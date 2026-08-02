"""Versioned RM65 pose-action to qpos lift profiles.

The pose-lift profile is intentionally independent from the human teleop
source profile.  A collected pose action has already absorbed tracker and
teleop source shaping; replay and policy deployment only own the inverse
mapping from that materialized pose to an executable qpos reference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from dataclasses import replace
from typing import Any


RM65_POSE_LIFT_PROFILE_SCHEMA = "robot_motion.rm65_pose_lift_profile/v1"
RM65_POSE_LIFT_V1_BASELINE_NAME = "rm65_pose_lift_v1_baseline"
RM65_POSE_LIFT_V2_CONTINUOUS_NAME = "rm65_pose_lift_v2_continuous"
RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME = (
    "rm65_pose_lift_v3_bounded_continuous"
)
RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME = "rm65_pose_lift_v4_qp_constrained"
RM65_POSE_LIFT_V5_SIGMA_ESCAPE_NAME = "rm65_pose_lift_v5_sigma_escape"
RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH_NAME = (
    "rm65_pose_lift_v6_structural_branch"
)
RM65_POSE_LIFT_V7_STRUCTURAL_ESCAPE_NAME = (
    "rm65_pose_lift_v7_structural_escape"
)
RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE_NAME = (
    "rm65_pose_lift_v8_nonlinear_best_candidate"
)
RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER_NAME = (
    "rm65_pose_lift_v9_predictive_branch_barrier"
)
RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD_NAME = (
    "rm65_pose_lift_v10_directional_feasibility_guard"
)


def _finite_non_negative(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


@dataclass(frozen=True)
class RM65PoseLiftProfile:
    """Stateful Pose->QPOS selection semantics, separate from hard safety.

    V1 is a named snapshot of the pre-profile deployment behavior.  V2 adds
    only pose-certified equivalent-solution normalization at this stage; it
    does not replay live-teleop smoothing or weaken any hard bound.
    """

    name: str
    algorithm_version: str
    equivalent_joint_unwrap: bool = False
    wrist_coaxial_canonicalization: bool = False
    wrist_soft_threshold_deg: float = 4.0
    wrist_hard_threshold_deg: float = 3.0
    wrist_null_motion_soft_limit_deg: float = 5.0
    wrist_bounded_local_search: bool = False
    wrist_local_search_scales: tuple[float, ...] = ()
    equivalence_translation_tolerance_m: float = 1e-3
    equivalence_rotation_tolerance_deg: float = 0.25
    target_translation_slack_m: float = 2.5e-4
    target_rotation_slack_deg: float = 0.05
    ik_posture_cost: float | None = None
    qp_substeps_per_reference: int = 0
    qp_velocity_limit: bool = False
    qp_acceleration_limit: bool = False
    qp_sigma_escape_cost: float = 0.0
    qp_sigma_escape_guard: float = 0.04
    qp_sigma_escape_rate_per_sec: float = 0.02
    qp_structural_branch_margin_deg: float = 0.0
    qp_structural_branch_guard_sigma: float = 0.04
    qp_structural_branch_release_sigma: float = 0.05
    qp_structural_branch_escape_cost: float = 0.0
    qp_structural_branch_escape_target_deg: float = 0.0
    qp_structural_branch_escape_guard_sigma: float = 0.04
    qp_structural_branch_constraint_mode: str = "hard_margin"
    qp_structural_branch_barrier_gain: float = 0.0
    qp_structural_branch_barrier_activation_sigma: float = 0.0
    qp_structural_branch_prediction_horizon_sec: float = 0.0
    directional_task_guard_max_utilization: float = 0.0
    directional_task_guard_enter_utilization: float = 1.0
    directional_task_guard_release_utilization: float = 0.75
    directional_task_guard_min_scale: float = 0.05
    directional_task_guard_structural_band_deg: float = 8.0
    directional_task_guard_release_band_deg: float = 10.0
    directional_task_guard_leaving_multiplier: float = 2.0
    qp_nonlinear_best_so_far: bool = False
    qp_nonlinear_best_min_iteration: int = 2
    qp_nonlinear_best_guard_sigma: float = 0.04
    qp_nonlinear_best_sigma_improvement: float = 1e-4
    qp_nonlinear_best_pose_score_slack: float = 0.05
    hardware_dispatch_allowed: bool = True

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        version = str(self.algorithm_version).strip()
        if not name or not version:
            raise ValueError("pose-lift profile name and algorithm_version must be non-empty.")
        soft = _finite_non_negative(
            self.wrist_soft_threshold_deg,
            name="wrist_soft_threshold_deg",
        )
        hard = _finite_non_negative(
            self.wrist_hard_threshold_deg,
            name="wrist_hard_threshold_deg",
        )
        null_limit = _finite_non_negative(
            self.wrist_null_motion_soft_limit_deg,
            name="wrist_null_motion_soft_limit_deg",
        )
        if hard > soft:
            raise ValueError("wrist_hard_threshold_deg must be <= wrist_soft_threshold_deg.")
        if null_limit <= 0.0:
            raise ValueError("wrist_null_motion_soft_limit_deg must be positive.")
        search_scales: list[float] = []
        for raw_scale in self.wrist_local_search_scales:
            scale = float(raw_scale)
            if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
                raise ValueError(
                    "wrist_local_search_scales must contain finite values in (0, 1]."
                )
            if not any(abs(scale - existing) <= 1e-12 for existing in search_scales):
                search_scales.append(scale)
        search_scales.sort(reverse=True)
        if self.wrist_bounded_local_search and not search_scales:
            raise ValueError(
                "wrist_local_search_scales must be non-empty when bounded search is enabled."
            )
        equivalence_translation = _finite_non_negative(
            self.equivalence_translation_tolerance_m,
            name="equivalence_translation_tolerance_m",
        )
        equivalence_rotation = _finite_non_negative(
            self.equivalence_rotation_tolerance_deg,
            name="equivalence_rotation_tolerance_deg",
        )
        target_translation_slack = _finite_non_negative(
            self.target_translation_slack_m,
            name="target_translation_slack_m",
        )
        target_rotation_slack = _finite_non_negative(
            self.target_rotation_slack_deg,
            name="target_rotation_slack_deg",
        )
        posture_cost = (
            None
            if self.ik_posture_cost is None
            else _finite_non_negative(
                self.ik_posture_cost,
                name="ik_posture_cost",
            )
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "algorithm_version", version)
        object.__setattr__(self, "equivalent_joint_unwrap", bool(self.equivalent_joint_unwrap))
        object.__setattr__(
            self,
            "wrist_coaxial_canonicalization",
            bool(self.wrist_coaxial_canonicalization),
        )
        object.__setattr__(self, "wrist_soft_threshold_deg", soft)
        object.__setattr__(self, "wrist_hard_threshold_deg", hard)
        object.__setattr__(self, "wrist_null_motion_soft_limit_deg", null_limit)
        object.__setattr__(
            self,
            "wrist_bounded_local_search",
            bool(self.wrist_bounded_local_search),
        )
        object.__setattr__(self, "wrist_local_search_scales", tuple(search_scales))
        object.__setattr__(
            self,
            "equivalence_translation_tolerance_m",
            equivalence_translation,
        )
        object.__setattr__(
            self,
            "equivalence_rotation_tolerance_deg",
            equivalence_rotation,
        )
        object.__setattr__(
            self,
            "target_translation_slack_m",
            target_translation_slack,
        )
        object.__setattr__(
            self,
            "target_rotation_slack_deg",
            target_rotation_slack,
        )
        object.__setattr__(self, "ik_posture_cost", posture_cost)
        substeps = int(self.qp_substeps_per_reference)
        if isinstance(self.qp_substeps_per_reference, bool) or substeps < 0:
            raise ValueError("qp_substeps_per_reference must be a non-negative integer.")
        object.__setattr__(self, "qp_substeps_per_reference", substeps)
        object.__setattr__(self, "qp_velocity_limit", bool(self.qp_velocity_limit))
        object.__setattr__(
            self,
            "qp_acceleration_limit",
            bool(self.qp_acceleration_limit),
        )
        object.__setattr__(
            self,
            "qp_sigma_escape_cost",
            _finite_non_negative(
                self.qp_sigma_escape_cost,
                name="qp_sigma_escape_cost",
            ),
        )
        object.__setattr__(
            self,
            "qp_sigma_escape_guard",
            _finite_non_negative(
                self.qp_sigma_escape_guard,
                name="qp_sigma_escape_guard",
            ),
        )
        object.__setattr__(
            self,
            "qp_sigma_escape_rate_per_sec",
            _finite_non_negative(
                self.qp_sigma_escape_rate_per_sec,
                name="qp_sigma_escape_rate_per_sec",
            ),
        )
        object.__setattr__(
            self,
            "qp_structural_branch_margin_deg",
            _finite_non_negative(
                self.qp_structural_branch_margin_deg,
                name="qp_structural_branch_margin_deg",
            ),
        )
        object.__setattr__(
            self,
            "qp_structural_branch_guard_sigma",
            _finite_non_negative(
                self.qp_structural_branch_guard_sigma,
                name="qp_structural_branch_guard_sigma",
            ),
        )
        object.__setattr__(
            self,
            "qp_structural_branch_release_sigma",
            _finite_non_negative(
                self.qp_structural_branch_release_sigma,
                name="qp_structural_branch_release_sigma",
            ),
        )
        object.__setattr__(
            self,
            "qp_structural_branch_escape_cost",
            _finite_non_negative(
                self.qp_structural_branch_escape_cost,
                name="qp_structural_branch_escape_cost",
            ),
        )
        object.__setattr__(
            self,
            "qp_structural_branch_escape_target_deg",
            _finite_non_negative(
                self.qp_structural_branch_escape_target_deg,
                name="qp_structural_branch_escape_target_deg",
            ),
        )
        object.__setattr__(
            self,
            "qp_structural_branch_escape_guard_sigma",
            _finite_non_negative(
                self.qp_structural_branch_escape_guard_sigma,
                name="qp_structural_branch_escape_guard_sigma",
            ),
        )
        branch_constraint_mode = str(
            self.qp_structural_branch_constraint_mode
        ).strip().lower()
        if branch_constraint_mode not in {
            "hard_margin",
            "barrier",
            "hybrid_deep_barrier",
        }:
            raise ValueError(
                "qp_structural_branch_constraint_mode must be "
                "'hard_margin', 'barrier' or 'hybrid_deep_barrier'."
            )
        object.__setattr__(
            self,
            "qp_structural_branch_constraint_mode",
            branch_constraint_mode,
        )
        object.__setattr__(
            self,
            "qp_structural_branch_barrier_gain",
            _finite_non_negative(
                self.qp_structural_branch_barrier_gain,
                name="qp_structural_branch_barrier_gain",
            ),
        )
        object.__setattr__(
            self,
            "qp_structural_branch_barrier_activation_sigma",
            _finite_non_negative(
                self.qp_structural_branch_barrier_activation_sigma,
                name="qp_structural_branch_barrier_activation_sigma",
            ),
        )
        object.__setattr__(
            self,
            "qp_structural_branch_prediction_horizon_sec",
            _finite_non_negative(
                self.qp_structural_branch_prediction_horizon_sec,
                name="qp_structural_branch_prediction_horizon_sec",
            ),
        )
        for field_name in (
            "directional_task_guard_max_utilization",
            "directional_task_guard_enter_utilization",
            "directional_task_guard_release_utilization",
            "directional_task_guard_min_scale",
            "directional_task_guard_structural_band_deg",
            "directional_task_guard_release_band_deg",
            "directional_task_guard_leaving_multiplier",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_non_negative(
                    getattr(self, field_name),
                    name=field_name,
                ),
            )
        if self.directional_task_guard_max_utilization > 0.0:
            if self.directional_task_guard_enter_utilization <= 0.0:
                raise ValueError(
                    "directional_task_guard_enter_utilization must be positive "
                    "when the directional guard is enabled."
                )
            if (
                self.directional_task_guard_release_utilization
                > self.directional_task_guard_enter_utilization
            ):
                raise ValueError(
                    "directional task guard release utilization must not "
                    "exceed its enter utilization."
                )
            if not 0.0 < self.directional_task_guard_min_scale <= 1.0:
                raise ValueError(
                    "directional_task_guard_min_scale must be in (0, 1]."
                )
            if (
                self.directional_task_guard_release_band_deg
                < self.directional_task_guard_structural_band_deg
            ):
                raise ValueError(
                    "directional task guard release band must be >= its "
                    "structural enter band."
                )
            if self.directional_task_guard_leaving_multiplier < 1.0:
                raise ValueError(
                    "directional_task_guard_leaving_multiplier must be >= 1."
                )
        object.__setattr__(
            self,
            "qp_nonlinear_best_so_far",
            bool(self.qp_nonlinear_best_so_far),
        )
        best_min_iteration = int(self.qp_nonlinear_best_min_iteration)
        if (
            isinstance(self.qp_nonlinear_best_min_iteration, bool)
            or best_min_iteration < 1
        ):
            raise ValueError(
                "qp_nonlinear_best_min_iteration must be a positive integer."
            )
        object.__setattr__(
            self,
            "qp_nonlinear_best_min_iteration",
            best_min_iteration,
        )
        object.__setattr__(
            self,
            "qp_nonlinear_best_guard_sigma",
            _finite_non_negative(
                self.qp_nonlinear_best_guard_sigma,
                name="qp_nonlinear_best_guard_sigma",
            ),
        )
        object.__setattr__(
            self,
            "qp_nonlinear_best_sigma_improvement",
            _finite_non_negative(
                self.qp_nonlinear_best_sigma_improvement,
                name="qp_nonlinear_best_sigma_improvement",
            ),
        )
        object.__setattr__(
            self,
            "qp_nonlinear_best_pose_score_slack",
            _finite_non_negative(
                self.qp_nonlinear_best_pose_score_slack,
                name="qp_nonlinear_best_pose_score_slack",
            ),
        )
        if self.qp_sigma_escape_cost > 0.0 and self.qp_sigma_escape_guard <= 0.0:
            raise ValueError("qp_sigma_escape_guard must be positive when escape is enabled.")
        if (
            self.qp_sigma_escape_cost > 0.0
            and self.qp_sigma_escape_rate_per_sec <= 0.0
        ):
            raise ValueError(
                "qp_sigma_escape_rate_per_sec must be positive when escape is enabled."
            )
        if self.qp_structural_branch_margin_deg > 0.0:
            if self.qp_structural_branch_guard_sigma <= 0.0:
                raise ValueError(
                    "qp_structural_branch_guard_sigma must be positive when branch guard is enabled."
                )
            if self.qp_structural_branch_release_sigma < self.qp_structural_branch_guard_sigma:
                raise ValueError(
                    "qp_structural_branch_release_sigma must be >= guard_sigma."
                )
            if (
                self.qp_structural_branch_constraint_mode
                in {"barrier", "hybrid_deep_barrier"}
                and self.qp_structural_branch_barrier_gain <= 0.0
            ):
                raise ValueError(
                    "qp_structural_branch_barrier_gain must be positive when "
                    "the branch constraint mode is 'barrier'."
                )
            if (
                self.qp_structural_branch_constraint_mode
                == "hybrid_deep_barrier"
                and not (
                    0.0
                    < self.qp_structural_branch_barrier_activation_sigma
                    <= self.qp_structural_branch_guard_sigma
                )
            ):
                raise ValueError(
                    "hybrid deep barrier activation sigma must be in "
                    "(0, guard_sigma]."
                )
        if self.qp_structural_branch_escape_cost > 0.0:
            if self.qp_structural_branch_margin_deg <= 0.0:
                raise ValueError(
                    "structural branch escape requires a positive one-sided margin."
                )
            if self.qp_structural_branch_escape_target_deg <= 0.0:
                raise ValueError(
                    "structural branch escape target must be positive when enabled."
                )
            if not (
                0.0 < self.qp_structural_branch_escape_guard_sigma
                <= self.qp_structural_branch_guard_sigma
            ):
                raise ValueError(
                    "structural branch escape guard sigma must be in (0, guard_sigma]."
                )
        if (
            self.qp_nonlinear_best_so_far
            and self.qp_nonlinear_best_guard_sigma <= 0.0
        ):
            raise ValueError(
                "qp_nonlinear_best_guard_sigma must be positive when "
                "best-so-far selection is enabled."
            )
        object.__setattr__(
            self,
            "hardware_dispatch_allowed",
            bool(self.hardware_dispatch_allowed),
        )

    def semantic_payload(self) -> dict[str, Any]:
        payload = {
            "schema": RM65_POSE_LIFT_PROFILE_SCHEMA,
            "name": self.name,
            "algorithm_version": self.algorithm_version,
            "equivalent_joint_unwrap": self.equivalent_joint_unwrap,
            "wrist_coaxial_canonicalization": self.wrist_coaxial_canonicalization,
            "wrist_soft_threshold_deg": self.wrist_soft_threshold_deg,
            "wrist_hard_threshold_deg": self.wrist_hard_threshold_deg,
            "wrist_null_motion_soft_limit_deg": self.wrist_null_motion_soft_limit_deg,
            "state_anchor": "last_formally_admitted_qref",
            "source_shaping_reapplied": False,
            "hard_safety_owned_by_profile": False,
        }
        # Preserve the exact V1/V2 metadata and fingerprints.  These fields
        # belong only to the opt-in V3 algorithm and are omitted from the
        # older frozen semantic snapshots.
        if self.wrist_bounded_local_search:
            payload.update(
                {
                    "wrist_bounded_local_search": True,
                    "wrist_local_search_scales": list(
                        self.wrist_local_search_scales
                    ),
                    "equivalence_translation_tolerance_m": (
                        self.equivalence_translation_tolerance_m
                    ),
                    "equivalence_rotation_tolerance_deg": (
                        self.equivalence_rotation_tolerance_deg
                    ),
                    "target_translation_slack_m": self.target_translation_slack_m,
                    "target_rotation_slack_deg": self.target_rotation_slack_deg,
                    "ik_posture_cost": self.ik_posture_cost,
                    "candidate_selection": (
                        "per_side_strongest_pose_equivalent_continuity_correction"
                    ),
                }
            )
        if (
            self.qp_substeps_per_reference > 0
            or self.qp_velocity_limit
            or self.qp_acceleration_limit
        ):
            payload.update(
                {
                    "qp_substeps_per_reference": self.qp_substeps_per_reference,
                    "qp_substep_time_contract": (
                        "none"
                        if self.qp_substeps_per_reference <= 0
                        else "reference_period_divided_by_substeps"
                    ),
                    "qp_velocity_limit": self.qp_velocity_limit,
                    "qp_acceleration_limit": self.qp_acceleration_limit,
                    "qp_previous_velocity_source": (
                        "last_formally_admitted_qref"
                    ),
                    "qp_iteration_constraint_scope": (
                        "final_candidate_relative_to_outer_admitted_anchor"
                    ),
                    "post_qp_joint_limit_role": "independent_admission_certificate",
                }
            )
        if self.qp_sigma_escape_cost > 0.0:
            payload.update(
                {
                    "qp_sigma_escape_cost": self.qp_sigma_escape_cost,
                    "qp_sigma_escape_guard": self.qp_sigma_escape_guard,
                    "qp_sigma_escape_rate_per_sec": (
                        self.qp_sigma_escape_rate_per_sec
                    ),
                    "qp_sigma_escape_metric": (
                        "normalized_geometric_jacobian_sigma_min"
                    ),
                    "qp_sigma_escape_characteristic_length_m": 0.5,
                    "qp_sigma_escape_gradient": (
                        "anchor_central_finite_difference_active_chain_only"
                    ),
                }
            )
        if self.qp_structural_branch_margin_deg > 0.0:
            payload.update(
                {
                    "qp_structural_branch_margin_deg": (
                        self.qp_structural_branch_margin_deg
                    ),
                    "qp_structural_branch_guard_sigma": (
                        self.qp_structural_branch_guard_sigma
                    ),
                    "qp_structural_branch_release_sigma": (
                        self.qp_structural_branch_release_sigma
                    ),
                    "qp_structural_branch_metric": "rm65_q3_q5_one_sided_zero_lock",
                    "qp_structural_branch_authority": (
                        "offline_only_soft_profile_hard_qp_inequality"
                    ),
                    "qp_structural_branch_escape_cost": (
                        self.qp_structural_branch_escape_cost
                    ),
                    "qp_structural_branch_escape_target_deg": (
                        self.qp_structural_branch_escape_target_deg
                    ),
                    "qp_structural_branch_escape_guard_sigma": (
                        self.qp_structural_branch_escape_guard_sigma
                    ),
                    "qp_structural_branch_escape_semantics": (
                        "soft_one_sided_target_from_admitted_anchor"
                    ),
                }
            )
            if (
                self.qp_structural_branch_constraint_mode != "hard_margin"
                or self.qp_structural_branch_barrier_gain > 0.0
                or self.qp_structural_branch_barrier_activation_sigma > 0.0
                or self.qp_structural_branch_prediction_horizon_sec > 0.0
            ):
                payload.update(
                    {
                        "qp_structural_branch_constraint_mode": (
                            self.qp_structural_branch_constraint_mode
                        ),
                        "qp_structural_branch_barrier_gain": (
                            self.qp_structural_branch_barrier_gain
                        ),
                        "qp_structural_branch_barrier_activation_sigma": (
                            self.qp_structural_branch_barrier_activation_sigma
                        ),
                        "qp_structural_branch_prediction_horizon_sec": (
                            self.qp_structural_branch_prediction_horizon_sec
                        ),
                        "qp_structural_branch_prediction_source": (
                            "last_formally_admitted_joint_velocity"
                        ),
                    }
                )
        if self.directional_task_guard_max_utilization > 0.0:
            payload.update(
                {
                    "directional_task_guard_max_utilization": (
                        self.directional_task_guard_max_utilization
                    ),
                    "directional_task_guard_enter_utilization": (
                        self.directional_task_guard_enter_utilization
                    ),
                    "directional_task_guard_release_utilization": (
                        self.directional_task_guard_release_utilization
                    ),
                    "directional_task_guard_min_scale": (
                        self.directional_task_guard_min_scale
                    ),
                    "directional_task_guard_structural_band_deg": (
                        self.directional_task_guard_structural_band_deg
                    ),
                    "directional_task_guard_release_band_deg": (
                        self.directional_task_guard_release_band_deg
                    ),
                    "directional_task_guard_leaving_multiplier": (
                        self.directional_task_guard_leaving_multiplier
                    ),
                    "directional_task_guard_metric": (
                        "weighted_pink_frame_task_svd_velocity_utilization"
                    ),
                    "directional_task_guard_target_shaping": (
                        "local_se3_log_singular_direction_projection"
                    ),
                    "directional_task_guard_authority": (
                        "offline_profile_target_shaping_before_pink_qp"
                    ),
                }
            )
        if self.qp_nonlinear_best_so_far:
            payload.update(
                {
                    "qp_nonlinear_best_so_far": True,
                    "qp_nonlinear_best_min_iteration": (
                        self.qp_nonlinear_best_min_iteration
                    ),
                    "qp_nonlinear_best_guard_sigma": (
                        self.qp_nonlinear_best_guard_sigma
                    ),
                    "qp_nonlinear_best_sigma_improvement": (
                        self.qp_nonlinear_best_sigma_improvement
                    ),
                    "qp_nonlinear_best_pose_score_slack": (
                        self.qp_nonlinear_best_pose_score_slack
                    ),
                    "qp_nonlinear_candidate_evaluation": (
                        "exact_fk_exact_normalized_jacobian_joint_and_collision_certificate"
                    ),
                    "qp_nonlinear_candidate_selection": (
                        "guarded_sigma_then_pose_else_pose_then_sigma"
                    ),
                }
            )
        if not self.hardware_dispatch_allowed:
            payload["hardware_dispatch_allowed"] = False
            payload["validation_scope"] = "offline_or_dry_run_only"
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.semantic_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_metadata(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "fingerprint": self.fingerprint}


RM65_POSE_LIFT_V1_BASELINE = RM65PoseLiftProfile(
    name=RM65_POSE_LIFT_V1_BASELINE_NAME,
    algorithm_version="current_deploy_snapshot_2026_07_28",
)

RM65_POSE_LIFT_V2_CONTINUOUS = RM65PoseLiftProfile(
    name=RM65_POSE_LIFT_V2_CONTINUOUS_NAME,
    algorithm_version="equivalent_solution_continuity_v1",
    equivalent_joint_unwrap=True,
    wrist_coaxial_canonicalization=True,
)

RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS = RM65PoseLiftProfile(
    name=RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME,
    algorithm_version="bounded_equivalent_continuity_v1",
    equivalent_joint_unwrap=True,
    wrist_coaxial_canonicalization=True,
    wrist_bounded_local_search=True,
    wrist_local_search_scales=(1.0, 0.75, 0.5, 0.25, 0.125, 0.0625),
    ik_posture_cost=0.0,
)

RM65_POSE_LIFT_V4_QP_CONSTRAINED = RM65PoseLiftProfile(
    name=RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
    algorithm_version="admitted_anchor_qp_velocity_acceleration_constraint_v1",
    equivalent_joint_unwrap=True,
    wrist_coaxial_canonicalization=True,
    wrist_bounded_local_search=True,
    wrist_local_search_scales=(1.0, 0.75, 0.5, 0.25, 0.125, 0.0625),
    ik_posture_cost=0.0,
    qp_substeps_per_reference=0,
    qp_velocity_limit=True,
    qp_acceleration_limit=True,
    hardware_dispatch_allowed=True,
)

RM65_POSE_LIFT_V5_SIGMA_ESCAPE = RM65PoseLiftProfile(
    name=RM65_POSE_LIFT_V5_SIGMA_ESCAPE_NAME,
    algorithm_version="admitted_anchor_qp_sigma_min_escape_v1",
    equivalent_joint_unwrap=True,
    wrist_coaxial_canonicalization=True,
    wrist_bounded_local_search=True,
    wrist_local_search_scales=(1.0, 0.75, 0.5, 0.25, 0.125, 0.0625),
    ik_posture_cost=0.0,
    qp_substeps_per_reference=0,
    qp_velocity_limit=True,
    qp_acceleration_limit=True,
    qp_sigma_escape_cost=0.15,
    qp_sigma_escape_guard=0.04,
    qp_sigma_escape_rate_per_sec=0.03,
    hardware_dispatch_allowed=False,
)

RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH = RM65PoseLiftProfile(
    name=RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH_NAME,
    algorithm_version="admitted_anchor_qp_sigma_escape_structural_branch_lock_v1",
    equivalent_joint_unwrap=True,
    wrist_coaxial_canonicalization=True,
    wrist_bounded_local_search=True,
    wrist_local_search_scales=(1.0, 0.75, 0.5, 0.25, 0.125, 0.0625),
    ik_posture_cost=0.0,
    qp_substeps_per_reference=0,
    qp_velocity_limit=True,
    qp_acceleration_limit=True,
    qp_sigma_escape_cost=0.15,
    qp_sigma_escape_guard=0.04,
    qp_sigma_escape_rate_per_sec=0.03,
    qp_structural_branch_margin_deg=0.75,
    qp_structural_branch_guard_sigma=0.04,
    qp_structural_branch_release_sigma=0.05,
    hardware_dispatch_allowed=False,
)

RM65_POSE_LIFT_V7_STRUCTURAL_ESCAPE = RM65PoseLiftProfile(
    name=RM65_POSE_LIFT_V7_STRUCTURAL_ESCAPE_NAME,
    algorithm_version="admitted_anchor_qp_sigma_escape_structural_branch_target_v1",
    equivalent_joint_unwrap=True,
    wrist_coaxial_canonicalization=True,
    wrist_bounded_local_search=True,
    wrist_local_search_scales=(1.0, 0.75, 0.5, 0.25, 0.125, 0.0625),
    ik_posture_cost=0.0,
    qp_substeps_per_reference=0,
    qp_velocity_limit=True,
    qp_acceleration_limit=True,
    qp_sigma_escape_cost=0.15,
    qp_sigma_escape_guard=0.04,
    qp_sigma_escape_rate_per_sec=0.03,
    qp_structural_branch_margin_deg=0.75,
    qp_structural_branch_guard_sigma=0.04,
    qp_structural_branch_release_sigma=0.05,
    qp_structural_branch_escape_cost=0.10,
    qp_structural_branch_escape_target_deg=4.0,
    qp_structural_branch_escape_guard_sigma=0.005,
    hardware_dispatch_allowed=False,
)

RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE = RM65PoseLiftProfile(
    name=RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE_NAME,
    algorithm_version="structural_branch_exact_nonlinear_best_so_far_v1",
    equivalent_joint_unwrap=True,
    wrist_coaxial_canonicalization=True,
    wrist_bounded_local_search=True,
    wrist_local_search_scales=(1.0, 0.75, 0.5, 0.25, 0.125, 0.0625),
    ik_posture_cost=0.0,
    qp_substeps_per_reference=0,
    qp_velocity_limit=True,
    qp_acceleration_limit=True,
    qp_sigma_escape_cost=0.15,
    qp_sigma_escape_guard=0.04,
    qp_sigma_escape_rate_per_sec=0.03,
    qp_structural_branch_margin_deg=0.75,
    qp_structural_branch_guard_sigma=0.04,
    qp_structural_branch_release_sigma=0.05,
    qp_nonlinear_best_so_far=True,
    qp_nonlinear_best_min_iteration=2,
    qp_nonlinear_best_guard_sigma=0.04,
    qp_nonlinear_best_sigma_improvement=1e-4,
    qp_nonlinear_best_pose_score_slack=0.05,
    hardware_dispatch_allowed=False,
)

RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER = RM65PoseLiftProfile(
    name=RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER_NAME,
    algorithm_version="predictive_structural_branch_cbf_v1",
    equivalent_joint_unwrap=True,
    wrist_coaxial_canonicalization=True,
    wrist_bounded_local_search=True,
    wrist_local_search_scales=(1.0, 0.75, 0.5, 0.25, 0.125, 0.0625),
    ik_posture_cost=0.0,
    qp_substeps_per_reference=0,
    qp_velocity_limit=True,
    qp_acceleration_limit=True,
    qp_sigma_escape_cost=0.15,
    qp_sigma_escape_guard=0.04,
    qp_sigma_escape_rate_per_sec=0.03,
    qp_structural_branch_margin_deg=0.75,
    qp_structural_branch_guard_sigma=0.04,
    qp_structural_branch_release_sigma=0.05,
    qp_structural_branch_constraint_mode="hybrid_deep_barrier",
    qp_structural_branch_barrier_gain=6.0,
    qp_structural_branch_barrier_activation_sigma=0.01,
    qp_structural_branch_prediction_horizon_sec=0.15,
    hardware_dispatch_allowed=False,
)

RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD = RM65PoseLiftProfile(
    name=RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD_NAME,
    algorithm_version="directional_feasibility_guard_v1",
    equivalent_joint_unwrap=True,
    wrist_coaxial_canonicalization=True,
    wrist_bounded_local_search=True,
    wrist_local_search_scales=(1.0, 0.75, 0.5, 0.25, 0.125, 0.0625),
    ik_posture_cost=0.0,
    qp_substeps_per_reference=0,
    qp_velocity_limit=True,
    qp_acceleration_limit=True,
    qp_structural_branch_margin_deg=0.25,
    qp_structural_branch_guard_sigma=0.04,
    qp_structural_branch_release_sigma=0.05,
    qp_structural_branch_constraint_mode="barrier",
    qp_structural_branch_barrier_gain=6.0,
    directional_task_guard_max_utilization=0.85,
    directional_task_guard_enter_utilization=1.0,
    directional_task_guard_release_utilization=0.75,
    directional_task_guard_min_scale=0.05,
    directional_task_guard_structural_band_deg=8.0,
    directional_task_guard_release_band_deg=10.0,
    directional_task_guard_leaving_multiplier=2.0,
    hardware_dispatch_allowed=False,
)


RM65_POSE_LIFT_PROFILES = {
    RM65_POSE_LIFT_V1_BASELINE.name: RM65_POSE_LIFT_V1_BASELINE,
    RM65_POSE_LIFT_V2_CONTINUOUS.name: RM65_POSE_LIFT_V2_CONTINUOUS,
    RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS.name: (
        RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS
    ),
    RM65_POSE_LIFT_V4_QP_CONSTRAINED.name: RM65_POSE_LIFT_V4_QP_CONSTRAINED,
    RM65_POSE_LIFT_V5_SIGMA_ESCAPE.name: RM65_POSE_LIFT_V5_SIGMA_ESCAPE,
    RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH.name: RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH,
    RM65_POSE_LIFT_V7_STRUCTURAL_ESCAPE.name: RM65_POSE_LIFT_V7_STRUCTURAL_ESCAPE,
    RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE.name: (
        RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE
    ),
    RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER.name: (
        RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER
    ),
    RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD.name: (
        RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD
    ),
}
RM65_POSE_LIFT_PROFILE_CHOICES = tuple(RM65_POSE_LIFT_PROFILES)


def resolve_rm65_pose_lift_profile(
    value: str | RM65PoseLiftProfile | None,
) -> RM65PoseLiftProfile:
    if isinstance(value, RM65PoseLiftProfile):
        return value
    name = str(value or RM65_POSE_LIFT_V1_BASELINE_NAME).strip()
    try:
        return RM65_POSE_LIFT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown RM65 pose-lift profile {value!r}; "
            f"expected one of {RM65_POSE_LIFT_PROFILE_CHOICES}."
        ) from exc


def build_rm65_pose_lift_ik_tuning(
    *,
    position_control_profile: str,
    pose_lift_profile: str | RM65PoseLiftProfile,
    orientation_weight: float,
) -> Any | None:
    """Build the shared replay/deploy IK objective for one Lift profile.

    Live-human source shaping is intentionally absent.  Returning ``None``
    for the explicit legacy position profile preserves its prior defaults.
    """

    from .ik_tuning import IKObjectiveWeights, load_ik_tuning_config
    from .position_profile import (
        LEGACY_POSITION_CONTROL_PROFILE_NAME,
        resolve_rm65_bimanual_position_profile,
    )

    position_name = str(position_control_profile).strip()
    if position_name == LEGACY_POSITION_CONTROL_PROFILE_NAME:
        return None
    position = resolve_rm65_bimanual_position_profile(position_name)
    lift = resolve_rm65_pose_lift_profile(pose_lift_profile)
    weight = float(orientation_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("orientation_weight must be finite and non-negative.")
    tuning = load_ik_tuning_config()
    mode_name = f"{lift.name}.ik_objective"
    position_weight = float(position.ik_execution.position_weight)
    objective_modes = dict(tuning.objective_modes)
    objective_modes[mode_name] = IKObjectiveWeights(
        single_arm_position_weight=position_weight,
        single_arm_orientation_weight=weight,
        dual_arm_position_cost=position_weight,
        dual_arm_orientation_cost=weight,
    )
    return replace(
        tuning,
        ik_objective_mode=mode_name,
        objective_modes=objective_modes,
    )


__all__ = [
    "RM65_POSE_LIFT_PROFILE_CHOICES",
    "RM65_POSE_LIFT_PROFILE_SCHEMA",
    "RM65_POSE_LIFT_PROFILES",
    "RM65_POSE_LIFT_V1_BASELINE",
    "RM65_POSE_LIFT_V1_BASELINE_NAME",
    "RM65_POSE_LIFT_V2_CONTINUOUS",
    "RM65_POSE_LIFT_V2_CONTINUOUS_NAME",
    "RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS",
    "RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME",
    "RM65_POSE_LIFT_V4_QP_CONSTRAINED",
    "RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME",
    "RM65_POSE_LIFT_V5_SIGMA_ESCAPE",
    "RM65_POSE_LIFT_V5_SIGMA_ESCAPE_NAME",
    "RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH",
    "RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH_NAME",
    "RM65_POSE_LIFT_V7_STRUCTURAL_ESCAPE",
    "RM65_POSE_LIFT_V7_STRUCTURAL_ESCAPE_NAME",
    "RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE",
    "RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE_NAME",
    "RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER",
    "RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER_NAME",
    "RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD",
    "RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD_NAME",
    "RM65PoseLiftProfile",
    "build_rm65_pose_lift_ik_tuning",
    "resolve_rm65_pose_lift_profile",
]

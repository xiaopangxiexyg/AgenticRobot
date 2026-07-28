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


RM65_POSE_LIFT_PROFILES = {
    RM65_POSE_LIFT_V1_BASELINE.name: RM65_POSE_LIFT_V1_BASELINE,
    RM65_POSE_LIFT_V2_CONTINUOUS.name: RM65_POSE_LIFT_V2_CONTINUOUS,
    RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS.name: (
        RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS
    ),
    RM65_POSE_LIFT_V4_QP_CONSTRAINED.name: RM65_POSE_LIFT_V4_QP_CONSTRAINED,
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
    "RM65PoseLiftProfile",
    "build_rm65_pose_lift_ik_tuning",
    "resolve_rm65_pose_lift_profile",
]

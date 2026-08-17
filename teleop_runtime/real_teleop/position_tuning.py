"""Position-mode IK and tracker-intent configuration projections."""

from __future__ import annotations

import argparse
from dataclasses import replace

from robot_motion.rm65.ik_tuning import (
    RM65IKTuningConfig,
    with_rm65_position_ik_objective,
)
from robot_motion.profiles import IKSafetyProfile, load_ik_safety_profiles

from ..sources.pose.vive_motion import TrackerMotionIntentConfig


def _impedance_bimanual_ik_safety_profile() -> IKSafetyProfile:
    """Return force-control-only bounds for degraded coupled-IK candidates.

    These are deliberately *not* a Cartesian output rate limiter.  Applying
    3 mm / 0.5 deg after IK at the outer target rate creates a multi-second
    target chase.  Here the values only constrain a local candidate that would
    otherwise regress from the currently safe coupled solution; normal target
    tracking still uses the coupled solver and execution joint-rate limits.
    """

    profiles = load_ik_safety_profiles()
    try:
        base_profile = profiles["dual_arm_teleop"]
    except KeyError as exc:  # pragma: no cover - guarded by shipped profiles
        raise RuntimeError("Missing required dual_arm_teleop IK safety profile.") from exc
    bounds = {
        "max_position_worsen_m": 0.003,
        "max_orientation_worsen_deg": 0.5,
        "dynamic_worsen_enabled": False,
        "scope": "impedance_bimanual_degraded_ik_candidate_only",
    }
    return replace(
        base_profile,
        name=f"{base_profile.name}_impedance_local_bounds",
        local_step_max_position_worsen_m=float(bounds["max_position_worsen_m"]),
        local_step_max_orientation_worsen_deg=float(bounds["max_orientation_worsen_deg"]),
        local_step_dynamic_worsen_enabled=False,
        metadata={
            **dict(base_profile.metadata),
            "impedance_bimanual_local_step_bounds": bounds,
        },
    )


def _position_ik_tuning_for_args(
    args: argparse.Namespace,
    tuning: RM65IKTuningConfig,
    *,
    arm_control_mode: str,
) -> RM65IKTuningConfig:
    orientation_weight = getattr(args, "position_ik_orientation_weight", None)
    if arm_control_mode != "position" or orientation_weight is None:
        return tuning
    weight = float(orientation_weight)
    return with_rm65_position_ik_objective(
        tuning,
        position_weight=1.0,
        orientation_weight=weight,
        mode_name="position_runtime_override",
    )


def _position_tracker_motion_intent_config(
    args: argparse.Namespace,
    *,
    arm_control_mode: str,
) -> TrackerMotionIntentConfig:
    config = TrackerMotionIntentConfig()
    if arm_control_mode != "position":
        return config
    scale = float(
        getattr(args, "position_inactive_enter_threshold_scale", 1.0)
    )
    return replace(
        config,
        enter_linear_speed_m_s=config.enter_linear_speed_m_s * scale,
        enter_angular_speed_rad_s=config.enter_angular_speed_rad_s * scale,
        enter_translation_span_m=config.enter_translation_span_m * scale,
        enter_rotation_span_rad=config.enter_rotation_span_rad * scale,
    )



__all__ = [
    "_impedance_bimanual_ik_safety_profile",
    "_position_ik_tuning_for_args",
    "_position_tracker_motion_intent_config",
]

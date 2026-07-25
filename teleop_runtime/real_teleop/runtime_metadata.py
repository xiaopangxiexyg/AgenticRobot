"""Read-only runtime configuration metadata for real teleoperation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

from ..position_control import RealArmExecutionConfig
from .control_profile import arm_control_mode


def arm_execution_profile_metadata(args: argparse.Namespace) -> dict[str, object]:
    metadata: dict[str, object] = {}
    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        metadata["arm_execution_profile"] = resolved.to_metadata()
    position_profile = getattr(args, "position_control_profile_metadata", None)
    if isinstance(position_profile, Mapping):
        metadata["position_control_profile"] = dict(position_profile)
    return metadata


def position_ik_iteration_metadata(args: argparse.Namespace) -> dict[str, object]:
    """Return position-only IK tuning settings for audit/config metadata."""
    position_mode = arm_control_mode(args) == "position"
    return {
        "position_ik_orientation_weight": (
            None
            if not position_mode
            else getattr(args, "position_ik_orientation_weight", None)
        ),
        "position_ik_solve_max_iterations": (
            int(getattr(args, "position_ik_solve_max_iterations", 2))
            if position_mode
            else None
        ),
        "position_ik_solve_min_iterations": (
            int(getattr(args, "position_ik_solve_min_iterations", 2))
            if position_mode
            else None
        ),
        "position_ik_damping": (
            float(getattr(args, "position_ik_damping", 0.03))
            if position_mode
            else None
        ),
        "position_ik_adaptive_iterations": (
            bool(getattr(args, "position_ik_adaptive_iterations", False))
            if position_mode
            else False
        ),
        "position_ik_iteration_reserve_ms": (
            float(getattr(args, "position_ik_iteration_reserve_ms", 5.0))
            if position_mode
            else None
        ),
        "position_ik_candidate_certification_reserve_ms": (
            float(
                getattr(
                    args,
                    "position_ik_candidate_certification_reserve_ms",
                    0.0,
                )
            )
            if position_mode
            else None
        ),
        # This applies only while selecting a local IK candidate. It does not
        # raise the final qref execution velocity cap.
        "position_ik_local_step_clip_scale": (
            float(getattr(args, "position_ik_local_step_clip_scale", 1.0))
            if position_mode
            else None
        ),
        "position_joint_velocity_limit_profile": (
            str(getattr(args, "position_joint_velocity_limit_profile", "rm65"))
            if position_mode
            else None
        ),
        "position_inactive_enter_threshold_scale": (
            float(getattr(args, "position_inactive_enter_threshold_scale", 1.0))
            if position_mode
            else None
        ),
        "position_prefer_non_wrist_singular_branch": (
            bool(
                getattr(
                    args,
                    "position_prefer_non_wrist_singular_branch",
                    True,
                )
            )
            if position_mode
            else None
        ),
        "position_prefer_non_elbow_singular_branch": (
            bool(
                getattr(
                    args,
                    "position_prefer_non_elbow_singular_branch",
                    True,
                )
            )
            if position_mode
            else None
        ),
        "position_prefer_non_shoulder_singular_branch": (
            bool(
                getattr(
                    args,
                    "position_prefer_non_shoulder_singular_branch",
                    False,
                )
            )
            if position_mode
            else None
        ),
        "position_singularity_candidate_certificate": (
            bool(
                getattr(
                    args,
                    "position_singularity_candidate_certificate",
                    False,
                )
            )
            if position_mode
            else None
        ),
        # Formal position teleop deliberately gives the local J3/J5
        # candidate classification priority over the vendor SDK boundary when
        # the exact final-qref certificate is enabled.  Force/replay paths do
        # not use this policy.
        "position_sdk_singularity_advisory": (
            bool(
                getattr(
                    args,
                    "position_singularity_candidate_certificate",
                    False,
                )
            )
            if position_mode
            else None
        ),
    }


__all__ = [
    "arm_execution_profile_metadata",
    "position_ik_iteration_metadata",
]

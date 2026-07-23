"""Frontend binding for the versioned RM65 position-control profile."""

from __future__ import annotations

from typing import Any, Sequence

from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.rm65 import (
    LEGACY_POSITION_CONTROL_PROFILE_NAME,
    RM65_BIMANUAL_POSITION_V1_NAME,
    resolve_rm65_bimanual_position_profile,
)


def argv_has_option(raw_argv: Sequence[str], *options: str) -> bool:
    return any(
        str(token) == option or str(token).startswith(f"{option}=")
        for token in raw_argv
        for option in options
    )


def apply_rm65_position_profile_defaults(
    args: Any,
    *,
    raw_argv: Sequence[str],
    mode: str,
    include_source_shaping: bool,
) -> None:
    """Apply formal defaults without overriding explicit compatibility flags.

    ``include_source_shaping`` is true only for a live human teleoperation
    producer.  Dataset replay and policy deployment consume an already-shaped
    main action and therefore never receive tracker/post-IK shaping defaults.
    """

    requested = str(
        getattr(args, "position_control_profile", RM65_BIMANUAL_POSITION_V1_NAME)
        or RM65_BIMANUAL_POSITION_V1_NAME
    ).strip()
    explicit_profile = argv_has_option(raw_argv, "--position-control-profile")
    if str(mode) != "position":
        if explicit_profile and requested != LEGACY_POSITION_CONTROL_PROFILE_NAME:
            raise ValueError(
                "--position-control-profile is position-only and cannot alter "
                "force/impedance control."
            )
        setattr(args, "position_control_profile_resolved", None)
        setattr(args, "position_control_profile_metadata", None)
        return
    if requested == LEGACY_POSITION_CONTROL_PROFILE_NAME:
        setattr(args, "position_control_profile_resolved", requested)
        setattr(
            args,
            "position_control_profile_metadata",
            {
                "name": requested,
                "compatibility_mode": True,
                "source_shaping_applied": bool(include_source_shaping),
            },
        )
        return

    profile = resolve_rm65_bimanual_position_profile(requested)
    ik = profile.ik_execution
    command = profile.command_execution
    source = profile.source_shaping
    explicit_overrides: list[str] = []

    def default(attr: str, value: Any, *options: str) -> None:
        if not hasattr(args, attr):
            return
        if argv_has_option(raw_argv, *options):
            explicit_overrides.append(attr)
            return
        setattr(args, attr, value)

    default("hz", profile.rates.sample_hz, "--hz")
    default(
        "position_reference_hz",
        profile.rates.reference_hz,
        "--position-reference-hz",
    )
    default("dual_arm_ik_hz", profile.rates.ik_hz, "--dual-arm-ik-hz")
    default("arm_control_hz", profile.rates.command_hz, "--arm-control-hz")
    default(
        "arm_canfd_follow",
        command.canfd_follow,
        "--arm-canfd-follow",
        "--no-arm-canfd-follow",
    )
    default(
        "position_reference_max_age_sec",
        command.reference_max_age_sec,
        "--position-reference-max-age-sec",
    )
    default(
        "arm_canfd_trajectory_mode",
        command.canfd_trajectory_mode,
        "--arm-canfd-trajectory-mode",
    )
    default("arm_canfd_radio", command.canfd_radio, "--arm-canfd-radio")
    default(
        "position_ik_orientation_weight",
        ik.orientation_weight,
        "--position-ik-orientation-weight",
    )
    default(
        "position_ik_solve_min_iterations",
        ik.solve_min_iterations,
        "--position-ik-solve-min-iterations",
    )
    default(
        "position_ik_solve_max_iterations",
        ik.solve_max_iterations,
        "--position-ik-solve-max-iterations",
        "--ik-solve-max-iterations",
        "--solve-max-iterations",
    )
    default("position_ik_damping", ik.damping, "--position-ik-damping")
    default(
        "position_ik_adaptive_iterations",
        ik.adaptive_iterations,
        "--position-ik-adaptive-iterations",
        "--no-position-ik-adaptive-iterations",
    )
    default(
        "position_ik_iteration_reserve_ms",
        ik.iteration_reserve_ms,
        "--position-ik-iteration-reserve-ms",
    )
    default(
        "position_ik_candidate_certification_reserve_ms",
        ik.candidate_certification_reserve_ms,
        "--position-ik-candidate-certification-reserve-ms",
    )
    default(
        "position_ik_local_step_clip_scale",
        ik.local_step_candidate_clip_scale,
        "--position-ik-local-step-clip-scale",
        "--position-ik-local-step-scale",
    )
    default(
        "position_joint_velocity_limit_profile",
        ik.joint_velocity_limit_profile,
        "--position-joint-velocity-limit-profile",
    )
    default(
        "position_prefer_non_wrist_singular_branch",
        ik.prefer_non_wrist_singular_branch,
        "--position-prefer-non-wrist-singular-branch",
        "--no-position-prefer-non-wrist-singular-branch",
    )
    default(
        "position_prefer_non_elbow_singular_branch",
        ik.prefer_non_elbow_singular_branch,
        "--position-prefer-non-elbow-singular-branch",
        "--no-position-prefer-non-elbow-singular-branch",
    )
    default(
        "position_prefer_non_shoulder_singular_branch",
        ik.prefer_non_shoulder_singular_branch,
        "--position-prefer-non-shoulder-singular-branch",
        "--no-position-prefer-non-shoulder-singular-branch",
    )
    default(
        "position_singularity_candidate_certificate",
        ik.require_singularity_candidate_certificate,
        "--position-singularity-candidate-certificate",
        "--no-position-singularity-candidate-certificate",
    )
    if hasattr(args, "position_qref_max_angvel") and not argv_has_option(
        raw_argv, "--position-qref-max-angvel"
    ):
        velocity_profile = load_joint_velocity_limit_profiles()[
            ik.joint_velocity_limit_profile
        ]
        import numpy as np

        args.position_qref_max_angvel = tuple(
            float(value)
            for value in np.deg2rad(velocity_profile.max_velocity_deg_s)
        )
    elif hasattr(args, "position_qref_max_angvel"):
        explicit_overrides.append("position_qref_max_angvel")

    default(
        "position_command_planner",
        command.planner,
        "--position-command-planner",
    )
    default("arm_max_angvel", command.max_velocity_rad_s, "--arm-max-angvel")
    default(
        "position_max_acceleration_rad_s2",
        command.max_acceleration_rad_s2,
        "--position-max-acceleration-rad-s2",
    )
    default(
        "position_target_velocity_filter_tau_sec",
        command.target_velocity_filter_tau_sec,
        "--position-target-velocity-filter-tau-sec",
    )
    default(
        "position_velocity_ramp_sec",
        command.velocity_ramp_sec,
        "--position-velocity-ramp-sec",
    )
    default(
        "position_max_command_step_deg",
        command.max_command_step_deg,
        "--position-max-command-step-deg",
    )
    default(
        "arm_follow_qpos_alpha",
        command.follow_qpos_alpha,
        "--arm-follow-qpos-alpha",
    )
    default(
        "position_degraded_speed_scale",
        command.degraded_speed_scale,
        "--position-degraded-speed-scale",
    )

    if include_source_shaping:
        default(
            "position_inactive_enter_threshold_scale",
            source.inactive_enter_threshold_scale,
            "--position-inactive-enter-threshold-scale",
        )
        default(
            "position_post_ik_qpos_smoothing_alpha",
            source.post_ik_qpos_smoothing_alpha,
            "--position-post-ik-qpos-smoothing-alpha",
        )
        default(
            "position_post_ik_qpos_smoothing_fast_alpha",
            source.post_ik_qpos_smoothing_fast_alpha,
            "--position-post-ik-qpos-smoothing-fast-alpha",
        )
        default(
            "position_post_ik_qpos_smoothing_adaptive",
            source.post_ik_qpos_smoothing_adaptive,
            "--position-post-ik-qpos-smoothing-adaptive",
            "--no-position-post-ik-qpos-smoothing-adaptive",
        )

    metadata = profile.to_metadata()
    metadata.update(
        {
            "source_shaping_applied": bool(include_source_shaping),
            "source_shaping_reapplied": False if not include_source_shaping else None,
            "explicit_overrides": sorted(set(explicit_overrides)),
        }
    )
    setattr(args, "position_control_profile_resolved", profile.name)
    setattr(args, "position_control_profile_metadata", metadata)


__all__ = ["apply_rm65_position_profile_defaults", "argv_has_option"]

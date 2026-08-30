"""Read-only runtime configuration metadata for real teleoperation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict

from ..position_control import RealArmExecutionConfig
from .control_profile import arm_control_mode
from .position_admittance_shadow import (
    POSITION_ADMITTANCE_ACCEPTED_FILTER_TAU_SEC,
    POSITION_ADMITTANCE_ACCEPTED_TRANSLATION_STIFFNESS_N_M,
    POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG,
    POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_RAMP_SEC,
    _resolve_position_admittance_response_configs,
    position_admittance_rotation_response_weight,
    position_admittance_translation_response_weight,
    resolve_position_admittance_profile,
)


def arm_execution_profile_metadata(args: argparse.Namespace) -> dict[str, object]:
    metadata: dict[str, object] = {}
    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        metadata["arm_execution_profile"] = resolved.to_metadata()
    position_profile = getattr(args, "position_control_profile_metadata", None)
    if isinstance(position_profile, Mapping):
        metadata["position_control_profile"] = dict(position_profile)
    return metadata


def position_admittance_response_metadata(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Describe the effective pre-IK response, including inactive parameters."""

    cached_profile = getattr(args, "_resolved_position_admittance_profile", None)
    if cached_profile is not None:
        profile = resolve_position_admittance_profile(args)
        active = profile.active
        enabled = profile.requested
        shadow = enabled and not active
        rotation_enabled = profile.rotation_requested
        translation_config = profile.translation_config
        rotational_config = profile.rotational_config
        rotation_profile = profile.rotation_profile
    else:
        # This standalone metadata API predates executable Active authority.
        # It must remain able to describe response tuning before an operator
        # selects an Active side; that read-only compatibility path cannot
        # authorize or install a target modifier.
        active = bool(getattr(args, "position_admittance_active", False))
        shadow = bool(getattr(args, "position_admittance_shadow", False))
        enabled = active or shadow
        rotation_active = bool(
            getattr(args, "position_admittance_rotation_active", False)
        )
        rotation_shadow = bool(
            getattr(args, "position_admittance_rotation_shadow", False)
        )
        rotation_enabled = rotation_active or rotation_shadow
        translation_config, rotational_config, rotation_profile = (
            _resolve_position_admittance_response_configs(
                args,
                active=active,
                shadow=shadow,
                rotation_active=rotation_active,
                rotation_shadow=rotation_shadow,
            )
        )
    translation_weight = position_admittance_translation_response_weight(args)
    rotation_weight = position_admittance_rotation_response_weight(args)

    translation_effective_compliance = None
    translation_effective_weight = None
    translation_physical_damping_applied = None
    if translation_config is not None:
        if translation_config.response_model == "profile_compliance":
            translation_effective_weight = (
                POSITION_ADMITTANCE_ACCEPTED_TRANSLATION_STIFFNESS_N_M
                / float(translation_config.stiffness_n_m[0])
            )
            translation_effective_compliance = [
                1.0 / float(value) for value in translation_config.stiffness_n_m
            ]
            translation_physical_damping_applied = False
        else:
            translation_physical_damping_applied = True
    rotation_effective_compliance = (
        None
        if rotational_config is None
        else [
            1.0 / float(value)
            for value in rotational_config.stiffness_nm_rad
        ]
    )

    return {
        "schema_version": "teleop.position_admittance_response/v1",
        "enabled": enabled,
        "mode": "active" if active else "shadow" if shadow else "disabled",
        "modifier_stage": "pre_ik_target",
        "ik_logic_modified": False,
        "weight_semantics": (
            "accepted_relative_compliance_multiplier_before_unchanged_hard_bounds"
        ),
        "translation": {
            "response_weight": translation_weight,
            "effective_response_weight": translation_effective_weight,
            "declared_weight_matches_effective": (
                None
                if translation_effective_weight is None
                else abs(translation_weight - translation_effective_weight)
                <= 1e-12
            ),
            "accepted_stiffness_n_m": (
                POSITION_ADMITTANCE_ACCEPTED_TRANSLATION_STIFFNESS_N_M
            ),
            "base_stiffness_input_n_m": float(
                getattr(
                    args,
                    "position_admittance_shadow_stiffness_n_m",
                    POSITION_ADMITTANCE_ACCEPTED_TRANSLATION_STIFFNESS_N_M,
                )
            ),
            "effective_compliance_m_n": translation_effective_compliance,
            "input_filter_tau_sec": (
                None
                if translation_config is None
                else float(translation_config.force_filter_tau_sec)
            ),
            "output_follow_tau_sec": (
                None
                if translation_config is None
                else float(translation_config.output_follow_tau_sec)
            ),
            "max_output_acceleration_m_s2": (
                None
                if translation_config is None
                else list(translation_config.max_output_acceleration_m_s2)
            ),
            "physical_damping_n_s_m_applied": (
                translation_physical_damping_applied
            ),
            "declared_physical_damping_n_s_m": (
                None
                if translation_config is None
                else list(translation_config.damping_n_s_m)
            ),
            "effective_config": (
                None if translation_config is None else asdict(translation_config)
            ),
        },
        "rotation": {
            "enabled": rotation_enabled,
            "profile": rotation_profile,
            "response_weight": rotation_weight,
            "accepted_filter_tau_sec": POSITION_ADMITTANCE_ACCEPTED_FILTER_TAU_SEC,
            "effective_compliance_rad_nm": rotation_effective_compliance,
            "input_filter_tau_sec": (
                None
                if rotational_config is None
                else float(rotational_config.torque_filter_tau_sec)
            ),
            "output_follow_tau_sec": (
                None
                if rotational_config is None
                else float(rotational_config.output_follow_tau_sec)
            ),
            "max_output_angular_acceleration_rad_s2": (
                None
                if rotational_config is None
                else list(
                    rotational_config.max_output_angular_acceleration_rad_s2
                )
            ),
            "physical_damping_term_applied": False,
            "effective_config": (
                None if rotational_config is None else asdict(rotational_config)
            ),
        },
        "unchanged_authoritative_boundaries": [
            "final_modifier_offset_and_slew_envelope",
            "shared_60hz_ik",
            "collision_admission",
            "joint_and_dynamics_admission",
            "position_force_guard",
        ],
    }


def position_ik_iteration_metadata(args: argparse.Namespace) -> dict[str, object]:
    """Return position-only IK tuning settings for audit/config metadata."""
    position_mode = arm_control_mode(args) == "position"
    return {
        "position_admittance_response": (
            position_admittance_response_metadata(args)
            if position_mode
            else None
        ),
        "position_ik_step_mode": (
            str(getattr(args, "position_ik_step_mode", "legacy_iterative"))
            if position_mode
            else None
        ),
        "position_ik_effective_physical_steps_per_tick": (
            1
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_ik_admitted_motion_limit_enabled": (
            bool(getattr(args, "position_ik_admitted_motion_limit", False))
            if position_mode
            else False
        ),
        "position_ik_motion_limit_anchor": (
            "last_formally_admitted_qref_qdot"
            if (
                position_mode
                and bool(
                    getattr(args, "position_ik_admitted_motion_limit", False)
                )
            )
            else None
        ),
        # The single-step position path also protects the final smoothed qref
        # from crossing the RM65 q5 wrist gate.  The threshold is derived from
        # the 3 deg SDK boundary plus the narrow 0.25 deg recovery margin used
        # by the singularity-aware position solver.
        "position_wrist_q5_escape_enabled": (
            bool(
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            if position_mode
            else False
        ),
        "position_wrist_q5_escape_threshold_deg": (
            3.25
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_wrist_coaxial_soft_threshold_deg": (
            4.0
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_wrist_coaxial_hard_threshold_deg": (
            3.0
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_wrist_q5_escape_margin_deg": (
            0.25
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_wrist_q5_hysteresis_release_deg": (
            float(
                getattr(
                    args,
                    "position_wrist_q5_hysteresis_release_deg",
                    6.0,
                )
            )
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_wrist_q5_hysteresis_requires_escape_latch": (
            bool(
                getattr(
                    args,
                    "position_wrist_q5_hysteresis_requires_escape_latch",
                    False,
                )
            )
            if position_mode
            else None
        ),
        "position_post_ik_qpos_normal_fast_path": (
            bool(getattr(args, "position_post_ik_qpos_normal_fast_path", False))
            if position_mode
            else None
        ),
        "position_post_ik_qpos_normal_fast_path_alpha": (
            float(
                getattr(args, "position_post_ik_qpos_normal_fast_path_alpha", 1.0)
            )
            if position_mode
            else None
        ),
        "position_post_ik_qpos_normal_fast_path_guard_deg": (
            float(
                getattr(
                    args,
                    "position_post_ik_qpos_normal_fast_path_guard_deg",
                    4.0,
                )
            )
            if position_mode
            else None
        ),
        "position_post_ik_qpos_normal_fast_path_quality_grace_sec": (
            float(
                getattr(
                    args,
                    "position_post_ik_qpos_normal_fast_path_quality_grace_sec",
                    0.0,
                )
            )
            if position_mode
            else None
        ),
        "position_admittance_active_qpos_fast_path": (
            bool(getattr(args, "position_admittance_active", False))
            if position_mode
            else None
        ),
        "position_admittance_active_envelope": (
            str(getattr(args, "position_admittance_active_envelope", "cautious"))
            if position_mode
            else None
        ),
        "position_admittance_active_max_tool_excursion_m": (
            float(
                getattr(
                    args,
                    "position_admittance_active_max_tool_excursion_m",
                    0.08,
                )
            )
            if position_mode
            else None
        ),
        "position_admittance_rotation_active": (
            bool(getattr(args, "position_admittance_rotation_active", False))
            if position_mode
            else None
        ),
        "position_admittance_combined_active": (
            bool(getattr(args, "position_admittance_combined_active", False))
            if position_mode
            else None
        ),
        "position_admittance_rotation_active_profile": (
            str(
                getattr(
                    args,
                    "position_admittance_rotation_active_profile",
                    "cautious-3deg",
                )
            )
            if position_mode
            else None
        ),
        "position_admittance_active_max_tool_rotation_excursion_deg": (
            float(
                getattr(
                    args,
                    "position_admittance_active_max_tool_rotation_excursion_deg",
                    3.5,
                )
            )
            if position_mode
            else None
        ),
        "position_admittance_active_qpos_fast_path_alpha": (
            float(POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA)
            if position_mode
            else None
        ),
        "position_admittance_rotation_active_qpos_fast_path_alpha": (
            float(POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA)
            if position_mode
            else None
        ),
        "position_admittance_rotation_release_qpos_fast_path_alpha": (
            float(POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_FAST_PATH_ALPHA)
            if position_mode
            else None
        ),
        "position_admittance_rotation_release_qpos_ramp_sec": (
            float(POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_RAMP_SEC)
            if position_mode
            else None
        ),
        "position_admittance_active_qpos_micro_reversal_hold_deg": (
            float(POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG)
            if position_mode
            else None
        ),
        "position_wrist_q5_hysteresis_relax_for_far_target": (
            True
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        # J3 uses the same narrow 0.25 deg margin as the SDK elbow gate.  The
        # broader 1 deg branch-selection margin remains available to legacy
        # callers, but formal single-step teleop must not enter escape mode
        # during ordinary 4--6 deg motion.
        "position_elbow_escape_threshold_deg": (
            3.25
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_elbow_escape_margin_deg": (
            0.25
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_singularity_escape_policy": (
            {
                "directional": True,
                "combined_j3_j5": True,
                "selection_priority": "singularity_depth_then_pose_progress",
                "elbow_margin_deg": 0.25,
            }
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_wrist_q5_hysteresis_far_target_position_m": (
            0.10
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_wrist_q5_hysteresis_far_target_orientation_deg": (
            20.0
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_wrist_singularity_smoothing_alpha": (
            max(
                float(getattr(args, "position_post_ik_qpos_smoothing_fast_alpha", 0.45)),
                0.65,
            )
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_wrist_null_motion_soft_limit_deg": (
            5.0
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_wrist_branch_switch_guard_deg": (
            float(
                getattr(
                    args,
                    "position_wrist_branch_switch_guard_deg",
                    12.0,
                )
            )
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
        "position_singularity_escape_wide_search": (
            {
                "enabled": True,
                "position_error_threshold_m": 0.10,
                "orientation_error_threshold_deg": 20.0,
                "radius_factors": [1.0, 2.0, 4.0],
                "max_position_worsen_m": 0.05,
                "max_orientation_worsen_deg": 18.0,
            }
            if (
                position_mode
                and str(
                    getattr(args, "position_ik_step_mode", "legacy_iterative")
                )
                == "single_step_singularity_aware"
            )
            else None
        ),
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
    "position_admittance_response_metadata",
    "position_ik_iteration_metadata",
]

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from robot_motion.rm65.ik_tuning import load_ik_tuning_config
from robot_motion.position_control import (
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_LINEAR,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
    POSITION_15_60_120_PROFILE,
    POSITION_45_150_PROFILE,
    POSITION_60_120_PROFILE,
    POSITION_PROFILE,
    PositionControlRates,
    SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP,
    SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.position_control import (
    ControlDebugConfig,
    PositionDeployExecutorConfig,
    resolve_control_debug_config,
    resolve_real_arm_execution_config,
)
from teleop_runtime.real_teleop.control_profile import (
    output_qpos_smoothing_singularity_alpha,
)
from teleop_runtime.real_teleop.runtime import _tactile_storage_capacity_summary


def _execution_args(**overrides):
    values = {
        "arm_execution_profile": None,
        "arm_control_mode": "impedance",
        "hz": 15.0,
        "position_reference_hz": 60.0,
        "dual_arm_ik_hz": None,
        "arm_control_hz": None,
        "arm_canfd_follow": None,
        "arm_canfd_follow_stream_hz": None,
        "position_post_ik_qpos_smoothing_alpha": None,
        "arm_follow_qpos_alpha": None,
        "arm_canfd_follow_warning_ratio": None,
        "arm_canfd_trajectory_mode": None,
        "arm_canfd_radio": None,
        "position_reference_max_age_sec": 0.2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _parse_real_teleop_args(argv: list[str]) -> argparse.Namespace:
    with patch.object(
        sys,
        "argv",
        ["prog", "--no-device-preflight", *argv],
    ):
        return real_teleop.parse_args()


def test_tcp_pose_envelope_cli_is_default_off_and_requires_explicit_active() -> None:
    defaults = _parse_real_teleop_args([])
    assert defaults.tcp_pose_envelope_profile == ""
    assert defaults.tcp_pose_envelope_active is False

    shadow = _parse_real_teleop_args(
        ["--tcp-pose-envelope-profile", "tcp-envelope.json"]
    )
    assert shadow.tcp_pose_envelope_profile == "tcp-envelope.json"
    assert shadow.tcp_pose_envelope_active is False

    active = _parse_real_teleop_args(
        [
            "--tcp-pose-envelope-profile",
            "tcp-envelope.json",
            "--tcp-pose-envelope-active",
        ]
    )
    assert active.tcp_pose_envelope_active is True

    with pytest.raises(SystemExit):
        _parse_real_teleop_args(["--tcp-pose-envelope-active"])


def test_default_common_position_profile_contract_has_independent_15_60_120_clocks() -> None:
    rates = POSITION_PROFILE.rates

    assert rates.sample_hz == 15.0
    assert rates.reference_hz == 60.0
    assert rates.ik_hz == 60.0
    assert rates.command_hz == 120.0
    assert rates.ordering_warnings == ()
    assert rates.to_metadata()["independent_clocks"] is True


def test_hil_position_profile_is_60_input_60_reference_120_command() -> None:
    rates = POSITION_60_120_PROFILE.rates

    assert rates.sample_hz == 60.0
    assert rates.reference_hz == 60.0
    assert rates.ik_hz == 60.0
    assert rates.command_hz == 120.0
    assert POSITION_60_120_PROFILE.validate_compose_target_hz(60.0) == 60.0
    with pytest.raises(ValueError, match="below the controller input rate"):
        POSITION_60_120_PROFILE.validate_compose_target_hz(15.0)


def test_real_executor_can_bind_the_hil_profile_without_copying_control_settings() -> None:
    config = PositionDeployExecutorConfig.from_position_profile(
        POSITION_60_120_PROFILE,
        debug_events_enabled=True,
    )

    assert (config.sample_hz, config.reference_hz, config.command_hz) == (
        60.0,
        60.0,
        120.0,
    )
    assert (
        POSITION_60_120_PROFILE.rates.reference_hz
        == POSITION_15_60_120_PROFILE.rates.reference_hz
    )
    assert (
        POSITION_60_120_PROFILE.rates.ik_hz
        == POSITION_15_60_120_PROFILE.rates.ik_hz
    )
    assert (
        POSITION_60_120_PROFILE.rates.command_hz
        == POSITION_15_60_120_PROFILE.rates.command_hz
    )
    assert config.debug_events_enabled is True


def test_real_teleop_velocity_planner_is_explicit_opt_in_with_point_one_second_ramp() -> None:
    defaults = _parse_real_teleop_args([])
    assert defaults.position_command_planner == POSITION_COMMAND_PLANNER_LINEAR
    assert defaults.position_velocity_ramp_sec == pytest.approx(0.10)
    assert defaults.position_degraded_speed_scale == pytest.approx(0.50)
    assert defaults.position_max_acceleration_rad_s2 == pytest.approx(
        (7.0, 7.0, 9.0, 15.0, 15.0, 20.0)
    )

    enabled = _parse_real_teleop_args(
        [
            "--arm-execution-profile",
            "position",
            "--position-command-planner",
            "acceleration_limited",
            "--position-velocity-ramp-sec",
            "0.08",
            "--position-degraded-speed-scale",
            "0.2",
            "--position-max-acceleration-rad-s2",
            "8,8,10,16,16,21",
        ]
    )
    assert (
        enabled.position_command_planner
        == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
    )
    assert enabled.position_velocity_ramp_sec == pytest.approx(0.08)
    assert enabled.position_degraded_speed_scale == pytest.approx(0.2)
    assert enabled.position_max_acceleration_rad_s2 == pytest.approx(
        (8.0, 8.0, 10.0, 16.0, 16.0, 21.0)
    )


def test_sample_aligned_jerk_profile_resolves_atomically_without_changing_v1() -> None:
    enabled = _parse_real_teleop_args(
        [
            "--position-control-profile",
            "rm65_bimanual_position_sample_aligned_jerk_v1",
            "--arm-control-mode",
            "position",
        ]
    )

    assert (
        enabled.position_command_planner
        == POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
    )
    assert enabled.hz == pytest.approx(15.0)
    assert enabled.position_reference_hz == pytest.approx(60.0)
    assert enabled.dual_arm_ik_hz == pytest.approx(60.0)
    assert enabled.arm_control_hz == pytest.approx(120.0)
    assert enabled.position_velocity_ramp_sec == pytest.approx(0.05)
    assert enabled.position_target_velocity_filter_tau_sec == pytest.approx(0.0)
    assert enabled.arm_follow_qpos_alpha == pytest.approx(1.0)

    rollback = _parse_real_teleop_args(
        [
            "--position-control-profile",
            "rm65_bimanual_position_v1",
            "--arm-control-mode",
            "position",
        ]
    )
    assert rollback.position_command_planner == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
    assert rollback.position_velocity_ramp_sec == pytest.approx(0.10)
    assert rollback.position_target_velocity_filter_tau_sec == pytest.approx(0.03)
    assert rollback.arm_follow_qpos_alpha == pytest.approx(0.9)


def test_sample_aligned_jerk_v2_resolves_convergence_and_governor_contract() -> None:
    enabled = _parse_real_teleop_args(
        [
            "--position-control-profile",
            "rm65_bimanual_position_sample_aligned_jerk_v2",
            "--arm-control-mode",
            "position",
        ]
    )

    assert (
        enabled.position_command_planner
        == POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
    )
    assert enabled.position_post_ik_qpos_normal_fast_path is True
    assert enabled.position_post_ik_qpos_normal_fast_path_alpha == pytest.approx(0.70)
    assert enabled.position_sample_aligned_scale_degraded_target is False
    assert enabled.position_sample_aligned_jerk_regularization == pytest.approx(1e-6)
    assert enabled.position_sample_aligned_jerk_delta_regularization == pytest.approx(
        1e-5
    )
    assert enabled.position_sample_aligned_terminal_velocity_regularization == 0.0
    assert enabled.position_sample_aligned_terminal_acceleration_regularization == 0.0
    assert enabled.position_inactive_execution_release_translation_static_m == (
        pytest.approx(0.0015)
    )
    assert enabled.position_inactive_execution_release_translation_full_m == (
        pytest.approx(0.012)
    )
    assert enabled.position_inactive_execution_release_rotation_static_deg == (
        pytest.approx(0.25)
    )
    assert enabled.position_inactive_execution_release_rotation_full_deg == (
        pytest.approx(2.0)
    )


def test_sample_aligned_jerk_v3_resolves_nullspace_smoothing_contract() -> None:
    enabled = _parse_real_teleop_args(
        [
            "--position-control-profile",
            "rm65_bimanual_position_sample_aligned_jerk_v3",
            "--arm-control-mode",
            "position",
        ]
    )

    assert (
        enabled.position_sample_aligned_reference_mode
        == "linear_hold_nullspace_smooth"
    )
    assert enabled.position_sample_aligned_observer_stop_threshold_deg == pytest.approx(
        0.01
    )
    assert enabled.position_sample_aligned_observer_stop_velocity_decay == pytest.approx(
        0.0
    )
    assert enabled.position_sample_aligned_observer_window == 4
    assert enabled.position_sample_aligned_observer_efficiency_power == pytest.approx(
        2.0
    )
    assert enabled.position_sample_aligned_observer_velocity_ema_alpha == pytest.approx(
        0.5
    )
    assert enabled.position_sample_aligned_observer_velocity_cap_scale == pytest.approx(
        0.5
    )
    assert enabled.position_sample_aligned_observer_coherence_min == pytest.approx(0.5)
    assert enabled.position_sample_aligned_observer_speed_guard_scale == pytest.approx(
        0.4
    )
    assert (
        enabled.position_sample_aligned_terminal_velocity_displacement_scale
        == pytest.approx(0.75)
    )
    assert (
        enabled.position_sample_aligned_initial_jerk_delta_regularization
        == pytest.approx(0.0)
    )


def test_balanced_60hz_jerk_profile_preserves_ik_and_separates_execution_clock() -> None:
    balanced = _parse_real_teleop_args(
        [
            "--position-control-profile",
            "rm65_bimanual_position_v2_balanced",
            "--arm-control-mode",
            "position",
        ]
    )
    enabled = _parse_real_teleop_args(
        [
            "--position-control-profile",
            "rm65_bimanual_position_v2_balanced_60hz_jerk_v1",
            "--arm-control-mode",
            "position",
        ]
    )
    optimized = _parse_real_teleop_args(
        [
            "--position-control-profile",
            "rm65_bimanual_position_v2_balanced_60hz_jerk_v2",
            "--arm-control-mode",
            "position",
        ]
    )

    assert enabled.hz == pytest.approx(15.0)
    assert enabled.position_reference_hz == pytest.approx(60.0)
    assert enabled.dual_arm_ik_hz == pytest.approx(60.0)
    assert enabled.arm_control_hz == pytest.approx(120.0)
    assert enabled.position_sample_aligned_source_hz == pytest.approx(60.0)
    assert enabled.position_sample_aligned_source_latch_mode == (
        SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET
    )
    assert optimized.position_sample_aligned_source_latch_mode == (
        SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP
    )
    assert enabled.position_sample_aligned_horizon_source_steps == 2
    assert enabled.position_sample_aligned_safety_horizon_source_steps == 20
    assert enabled.position_command_planner == (
        POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
    )
    assert enabled.position_sample_aligned_jerk_delta_regularization == pytest.approx(
        1e-4
    )
    for attr in (
        "position_ik_orientation_weight",
        "position_ik_solve_min_iterations",
        "position_ik_solve_max_iterations",
        "position_ik_damping",
        "position_ik_adaptive_iterations",
        "position_ik_iteration_reserve_ms",
        "position_ik_candidate_certification_reserve_ms",
        "position_ik_local_step_clip_scale",
        "position_joint_velocity_limit_profile",
        "position_post_ik_qpos_normal_fast_path_alpha",
    ):
        assert getattr(enabled, attr) == getattr(balanced, attr)
        assert getattr(optimized, attr) == getattr(balanced, attr)


def test_formal_collection_process_is_position_auto_only_with_thread_rollback() -> None:
    impedance = _parse_real_teleop_args([])
    assert impedance.collection_worker_mode == "auto"
    assert impedance.collection_worker_mode_resolved == "thread"

    position = _parse_real_teleop_args(
        ["--arm-execution-profile", "position"]
    )
    assert position.collection_worker_mode_resolved == "process"
    assert position.collection_process_queue_size == 4
    assert position.collection_process_shared_slots == 4
    assert position.collection_process_extra_array_mb == 8
    assert position.collection_worker_nice == 5

    rollback = _parse_real_teleop_args(
        [
            "--arm-execution-profile",
            "position",
            "--collection-worker-mode",
            "thread",
        ]
    )
    assert rollback.collection_worker_mode_resolved == "thread"

    with pytest.raises(SystemExit):
        _parse_real_teleop_args(["--collection-worker-mode", "process"])


def test_dmtac_tactile_worker_cpu_affinity_accepts_two_core_profile() -> None:
    args = _parse_real_teleop_args(
        ["--dmtac-tactile-worker-cpus", "12,13"]
    )

    assert args.dmtac_tactile_worker_cpus == (12, 13)


def test_dataset_root_rejects_explicit_empty_string() -> None:
    with pytest.raises(SystemExit):
        _parse_real_teleop_args(["--dataset-root", ""])


def test_position_admittance_requires_coupled_dual_arm_topology(
    capsys: pytest.CaptureFixture[str],
) -> None:
    common = [
        "--position-control-profile",
        "rm65_bimanual_position_v1",
        "--arm-control-mode",
        "position",
        "--position-force-guard",
        "--position-force-guard-source",
        "tool",
    ]

    shadow = _parse_real_teleop_args(
        [*common, "--active-sides", "both", "--position-admittance-shadow"]
    )
    assert shadow.active_sides == ("left", "right")
    assert shadow.position_admittance_shadow is True
    assert shadow._resolved_position_admittance_profile.mode == "shadow"
    assert shadow._resolved_position_admittance_profile.rotational_config is None

    rotation_shadow = _parse_real_teleop_args(
        [
            *common,
            "--active-sides",
            "both",
            "--position-admittance-shadow",
            "--position-admittance-rotation-shadow",
            "--position-admittance-rotation-shadow-profile",
            "balanced-5deg",
        ]
    )
    assert rotation_shadow.position_admittance_rotation_shadow is True
    assert (
        rotation_shadow.position_admittance_rotation_shadow_profile
        == "balanced-5deg"
    )
    rotation_shadow_profile = rotation_shadow._resolved_position_admittance_profile
    assert rotation_shadow_profile.mode == "rotation_shadow"
    assert rotation_shadow_profile.translation_config.enabled_axes == (
        False,
        False,
        False,
    )
    assert rotation_shadow_profile.rotational_config is not None

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *common,
                "--active-sides",
                "both",
                "--position-admittance-shadow",
                "--position-admittance-translation-response-weight",
                "2",
            ]
        )
    assert "requires profile_compliance" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *common,
                "--active-sides",
                "both",
                "--position-admittance-rotation-shadow",
            ]
        )
    assert "requires --position-admittance-shadow" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *common,
                "--active-sides",
                "both",
                "--position-admittance-rotation-shadow-profile",
                "balanced-5deg",
            ]
        )
    assert "requires --position-admittance-rotation-shadow" in capsys.readouterr().err

    cautious_active_args = [
        *common,
        "--active-sides",
        "both",
        "--position-admittance-active",
        "--position-admittance-active-side",
        "left",
        "--position-admittance-response-model",
        "profile_compliance",
        "--position-admittance-shadow-max-offset-m",
        "0.002",
        "--position-admittance-shadow-max-velocity-m-s",
        "0.005",
        "--position-admittance-shadow-max-return-velocity-m-s",
        "0.005",
        "--position-admittance-shadow-max-slew-m-s",
        "0.005",
        "--position-admittance-shadow-stale-sec",
        "0.15",
        "--position-admittance-shadow-max-force-dt-sec",
        "0.15",
    ]
    active = _parse_real_teleop_args(cautious_active_args)
    assert active.position_admittance_active is True
    assert active.position_admittance_active_side == "left"
    assert active.position_admittance_active_envelope == "cautious"
    assert (
        active.position_admittance_active_endpoint_policy
        == "runtime_home_hold"
    )
    assert active.position_admittance_active_max_tool_excursion_m == pytest.approx(
        0.08
    )
    active_profile = active._resolved_position_admittance_profile
    assert active_profile.mode == "active"
    assert active_profile.apply_sides == ("left",)
    assert active_profile.max_tool_excursion_m == pytest.approx(0.08)

    rotation_active_args = [
        *cautious_active_args,
        "--position-admittance-rotation-active",
        "--position-admittance-rotation-active-profile",
        "cautious-3deg",
        "--position-admittance-active-max-tool-rotation-excursion-deg",
        "3.5",
    ]
    rotation_active = _parse_real_teleop_args(rotation_active_args)
    assert rotation_active.position_admittance_rotation_active is True
    assert rotation_active.position_admittance_combined_active is False
    assert (
        rotation_active.position_admittance_rotation_active_profile
        == "cautious-3deg"
    )
    assert (
        rotation_active.position_admittance_active_max_tool_rotation_excursion_deg
        == pytest.approx(3.5)
    )
    assert (
        rotation_active._resolved_position_admittance_profile.mode
        == "rotation_active"
    )

    combined_active = _parse_real_teleop_args(
        [*rotation_active_args, "--position-admittance-combined-active"]
    )
    assert combined_active.position_admittance_rotation_active is True
    assert combined_active.position_admittance_combined_active is True
    assert (
        combined_active._resolved_position_admittance_profile.mode
        == "combined_active"
    )

    normal_combined_args = [
        *rotation_active_args,
        "--position-admittance-combined-active",
        "--position-admittance-active-envelope",
        "normal_parity",
        "--position-admittance-shadow-max-offset-m",
        "0.08",
        "--position-admittance-shadow-max-velocity-m-s",
        "0.06",
        "--position-admittance-shadow-max-return-velocity-m-s",
        "0.42",
        "--position-admittance-shadow-max-slew-m-s",
        "0.06",
        "--position-admittance-shadow-return-rate-s",
        "4",
        "--position-admittance-rotation-active-profile",
        "balanced-5deg",
        "--position-admittance-active-max-tool-rotation-excursion-deg",
        "5.5",
    ]
    normal_combined = _parse_real_teleop_args(normal_combined_args)
    assert normal_combined.position_admittance_active_envelope == "normal_parity"
    assert normal_combined.position_admittance_shadow_max_offset_m == pytest.approx(
        0.08
    )
    assert normal_combined.position_admittance_shadow_max_velocity_m_s == pytest.approx(
        0.06
    )
    assert normal_combined.position_admittance_rotation_active_profile == (
        "balanced-5deg"
    )
    assert (
        normal_combined.position_admittance_active_max_tool_rotation_excursion_deg
        == pytest.approx(5.5)
    )

    formal_teleop = _parse_real_teleop_args(
        [
            *normal_combined_args,
            "--position-admittance-active-endpoint-policy",
            "formal_teleop",
        ]
    )
    assert (
        formal_teleop.position_admittance_active_endpoint_policy
        == "formal_teleop"
    )

    dual_formal_args = list(normal_combined_args)
    active_side_index = dual_formal_args.index(
        "--position-admittance-active-side"
    )
    dual_formal_args[active_side_index + 1] = "both"
    dual_formal = _parse_real_teleop_args(
        [
            *dual_formal_args,
            "--position-admittance-active-endpoint-policy",
            "formal_teleop",
        ]
    )
    assert dual_formal.position_admittance_active_side == "both"
    assert dual_formal.fix_arm is True

    five_pedal_formal = _parse_real_teleop_args(
        [
            *dual_formal_args,
            "--position-admittance-active-endpoint-policy",
            "formal_teleop",
            "--special-arm-control-mode",
            "foot-pad",
        ]
    )
    assert five_pedal_formal.fix_arm is True
    assert five_pedal_formal.special_arm_control_mode == "foot-pad"

    tuned_dual_formal_args = [
        *dual_formal_args,
        "--position-admittance-active-endpoint-policy",
        "formal_teleop",
        "--position-admittance-translation-response-weight",
        "1.25",
        "--position-admittance-rotation-response-weight",
        "1.75",
        "--position-admittance-translation-filter-tau-sec",
        "0.03",
        "--position-admittance-rotation-filter-tau-sec",
        "0.04",
    ]
    tuned_dual_formal = _parse_real_teleop_args(tuned_dual_formal_args)
    tuning_metadata = real_teleop._position_ik_iteration_metadata(
        tuned_dual_formal
    )["position_admittance_response"]
    assert tuning_metadata["translation"]["response_weight"] == pytest.approx(
        1.25
    )
    assert tuning_metadata["rotation"]["response_weight"] == pytest.approx(1.75)
    assert tuning_metadata["translation"]["input_filter_tau_sec"] == pytest.approx(
        0.03
    )
    assert tuning_metadata["rotation"]["input_filter_tau_sec"] == pytest.approx(
        0.04
    )

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *tuned_dual_formal_args,
                "--position-admittance-rotation-response-weight",
                "4.01",
            ]
        )
    assert "response weight must be in [0.25, 4]" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *tuned_dual_formal_args,
                "--position-admittance-translation-filter-tau-sec",
                "0.101",
            ]
        )
    assert "translation filter tau must be <= 0.1 s" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args([*dual_formal_args, "--no-fix-arm"])
    assert "dual-side Active requires formal_teleop" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *rotation_active_args,
                "--position-admittance-active-endpoint-policy",
                "formal_teleop",
            ]
        )
    assert "requires exact combined normal_parity" in capsys.readouterr().err

    emphasis_args = [
        (
            "rotation-emphasis-5deg"
            if value == "balanced-5deg"
            else value
        )
        for value in normal_combined_args
    ]
    emphasis = _parse_real_teleop_args(emphasis_args)
    assert emphasis.position_admittance_rotation_active_profile == (
        "rotation-emphasis-5deg"
    )

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *normal_combined_args,
                "--no-position-admittance-combined-active",
            ]
        )
    assert "requires explicit combined" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *normal_combined_args,
                "--position-admittance-active-max-tool-rotation-excursion-deg",
                "5.51",
            ]
        )
    assert "must be <= 5.5" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [*cautious_active_args, "--position-admittance-combined-active"]
        )
    assert "requires both" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *common,
                "--active-sides",
                "both",
                "--position-admittance-rotation-active",
            ]
        )
    assert "requires --position-admittance-active" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *rotation_active_args,
                "--position-admittance-active-max-tool-rotation-excursion-deg",
                "3.51",
            ]
        )
    assert "must be <= 3.5" in capsys.readouterr().err

    half_active_args = [
        *cautious_active_args,
        "--position-admittance-active-envelope",
        "half_normal",
        "--position-admittance-shadow-max-offset-m",
        "0.04",
        "--position-admittance-shadow-max-velocity-m-s",
        "0.03",
        "--position-admittance-shadow-max-return-velocity-m-s",
        "0.06",
        "--position-admittance-shadow-max-slew-m-s",
        "0.03",
        "--position-admittance-shadow-return-rate-s",
        "2",
        "--position-admittance-active-max-tool-excursion-m",
        "0.08",
    ]
    half_active = _parse_real_teleop_args(half_active_args)
    assert half_active.position_admittance_active_envelope == "half_normal"
    assert half_active.position_admittance_shadow_max_offset_m == pytest.approx(0.04)
    assert half_active.position_admittance_active_max_tool_excursion_m == pytest.approx(
        0.08
    )

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *half_active_args,
                "--position-admittance-active-max-tool-excursion-m",
                "0.081",
            ]
        )
    assert "max tool excursion must be <= 0.08" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *half_active_args,
                "--position-admittance-shadow-max-offset-m",
                "0.08",
            ]
        )
    assert "max_offset_m must be <= 0.04" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _parse_real_teleop_args(
            [
                *cautious_active_args,
                "--position-admittance-shadow-max-offset-m",
                "0.08",
            ]
        )
    assert "max_offset_m must be <= 0.002" in capsys.readouterr().err

    for unsupported in (
        ["--active-sides", "left", "--position-admittance-shadow"],
        ["--active-sides", "right", "--position-admittance-shadow"],
        [
            "--active-sides",
            "left",
            "--position-admittance-active",
            "--position-admittance-active-side",
            "left",
        ],
        [
            "--active-sides",
            "both",
            "--no-collision",
            "--position-admittance-shadow",
        ],
    ):
        with pytest.raises(SystemExit):
            _parse_real_teleop_args([*common, *unsupported])


def test_idle_tactile_compression_is_opt_in_and_rejects_hot_path_conflicts() -> None:
    defaults = _parse_real_teleop_args([])
    assert defaults.idle_tactile_compression is False
    enabled = _parse_real_teleop_args(["--idle-tactile-compression"])
    assert enabled.idle_tactile_compression is True
    rollback = _parse_real_teleop_args(
        ["--idle-tactile-compression", "--no-idle-tactile-compression"]
    )
    assert rollback.idle_tactile_compression is False

    combined = _parse_real_teleop_args(
        ["--idle-tactile-compression", "--tactile-artifact-integrity"]
    )
    assert combined.idle_tactile_compression is True
    assert combined.tactile_artifact_integrity is True

    for conflicting in (
        ["--idle-tactile-compression", "--no-collection"],
        ["--idle-tactile-compression", "--tactile-storage", "parquet"],
        ["--idle-tactile-compression", "--tactile-compression", "zstd"],
    ):
        with pytest.raises(SystemExit):
            _parse_real_teleop_args(conflicting)


def test_native_tactile_bundle_cli_is_opt_in_and_fail_closed(tmp_path: Path) -> None:
    defaults = _parse_real_teleop_args([])
    assert defaults.tactile_bundle_root == ""
    assert defaults.tactile_bundle_min_free_gb == 0.0

    tactile_root = tmp_path / "tactile-device"
    tactile_root.mkdir()
    enabled = _parse_real_teleop_args(
        [
            "--tactile-bundle-root",
            str(tactile_root),
            "--tactile-bundle-min-free-gb",
            "12.5",
        ]
    )
    assert enabled.tactile_bundle_root == str(tactile_root)
    assert enabled.tactile_bundle_min_free_gb == 12.5
    assert enabled.tactile_storage == "external"
    assert enabled.tactile_artifact_integrity is True

    for conflicting in (
        ["--tactile-bundle-min-free-gb", "1"],
        ["--tactile-bundle-root", "/tmp/tactile", "--no-collection"],
        [
            "--tactile-bundle-root",
            "/tmp/tactile",
            "--tactile-storage",
            "parquet",
        ],
        [
            "--tactile-bundle-root",
            "/tmp/tactile",
            "--no-tactile-artifact-integrity",
        ],
    ):
        with pytest.raises(SystemExit):
            _parse_real_teleop_args(conflicting)

    native_idle = _parse_real_teleop_args(
        [
            "--tactile-bundle-root",
            str(tactile_root),
            "--idle-tactile-compression",
            "--tactile-compression",
            "none",
        ]
    )
    assert native_idle.idle_tactile_compression is True
    assert native_idle.tactile_compression == "none"


def test_native_tactile_bundle_cli_reads_tactile_bundle_root_environment(
    monkeypatch, tmp_path: Path
) -> None:
    tactile_root = tmp_path / "env-tactile"
    tactile_root.mkdir()
    monkeypatch.setenv("TACTILE_BUNDLE_ROOT", str(tactile_root))

    enabled = _parse_real_teleop_args([])

    assert enabled.tactile_bundle_root == str(tactile_root)
    assert enabled.tactile_artifact_integrity is True


def test_native_tactile_bundle_cli_reads_minimum_free_space_environment(
    monkeypatch, tmp_path: Path
) -> None:
    tactile_root = tmp_path / "env-tactile"
    tactile_root.mkdir()
    monkeypatch.setenv("TACTILE_BUNDLE_ROOT", str(tactile_root))
    monkeypatch.setenv("TACTILE_BUNDLE_MIN_FREE_GB", "2")

    from_environment = _parse_real_teleop_args([])
    overridden = _parse_real_teleop_args(
        ["--tactile-bundle-min-free-gb", "3.5"]
    )

    assert from_environment.tactile_bundle_min_free_gb == 2.0
    assert from_environment.tactile_bundle_min_free_source == "environment"
    assert overridden.tactile_bundle_min_free_gb == 3.5
    assert overridden.tactile_bundle_min_free_source == "command_line"


def test_tactile_storage_capacity_summary_exposes_effective_reserve_source(
    monkeypatch, tmp_path: Path
) -> None:
    usage = SimpleNamespace(free=137_187_553_280)
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.runtime.shutil.disk_usage",
        lambda _root: usage,
    )
    args = SimpleNamespace(
        tactile_bundle_root=str(tmp_path),
        tactile_bundle_min_free_gb=200.0,
        tactile_bundle_min_free_source="environment",
    )

    summary = _tactile_storage_capacity_summary(args)

    assert summary is not None
    assert "free_gib=127.8" in summary
    assert "minimum_free_gib=200" in summary
    assert "source=environment" in summary
    assert "capacity=insufficient" in summary


def test_real_teleop_velocity_planner_rejects_non_position_or_disabled_stream() -> None:
    with pytest.raises(SystemExit):
        _parse_real_teleop_args(
            ["--position-command-planner", "acceleration_limited"]
        )


    with pytest.raises(SystemExit):
        _parse_real_teleop_args(
            [
                "--arm-execution-profile",
                "position",
                "--no-arm-canfd-follow",
                "--position-command-planner",
                "acceleration_limited",
            ]
        )
    with pytest.raises(SystemExit):
        _parse_real_teleop_args(
            [
                "--arm-execution-profile",
                "position",
                "--position-reference-hz",
                "100",
                "--arm-control-hz",
                "50",
                "--position-command-planner",
                "acceleration_limited",
            ]
        )


def test_real_teleop_position_ik_iteration_options_are_exposed_and_position_only() -> None:
    defaults = _parse_real_teleop_args([])
    assert defaults.position_ik_orientation_weight is None
    assert defaults.position_ik_solve_max_iterations == 2
    assert defaults.position_ik_solve_min_iterations == 2
    assert defaults.position_ik_damping == pytest.approx(0.03)
    assert defaults.position_ik_adaptive_iterations is False
    assert defaults.position_ik_iteration_reserve_ms == pytest.approx(5.0)
    assert defaults.position_ik_candidate_certification_reserve_ms == pytest.approx(0.0)
    assert defaults.position_singularity_candidate_certificate is False
    assert defaults.position_ik_local_step_clip_scale == pytest.approx(1.0)
    assert defaults.position_joint_velocity_limit_profile == "rm65"
    assert defaults.position_inactive_enter_threshold_scale == pytest.approx(1.0)

    tuned = _parse_real_teleop_args(
        [
            "--arm-execution-profile",
            "position",
            "--position-ik-orientation-weight",
            "0.5",
            "--position-ik-solve-max-iterations",
            "4",
            "--position-ik-solve-min-iterations",
            "3",
            "--position-ik-damping",
            "0.04",
            "--position-ik-adaptive-iterations",
            "--position-ik-iteration-reserve-ms",
            "6.5",
            "--position-ik-candidate-certification-reserve-ms",
            "3.0",
            "--position-singularity-candidate-certificate",
            "--position-ik-local-step-clip-scale",
            "2.5",
            "--position-inactive-enter-threshold-scale",
            "0.5",
            "--position-joint-velocity-limit-profile",
            "rm65_position_2x",
        ]
    )
    assert tuned.position_ik_orientation_weight == pytest.approx(0.5)
    assert tuned.position_ik_solve_max_iterations == 4
    assert tuned.position_ik_solve_min_iterations == 3
    assert tuned.position_ik_damping == pytest.approx(0.04)
    assert tuned.position_ik_adaptive_iterations is True
    assert tuned.position_ik_iteration_reserve_ms == pytest.approx(6.5)
    assert tuned.position_ik_candidate_certification_reserve_ms == pytest.approx(3.0)
    assert tuned.position_singularity_candidate_certificate is True
    assert tuned.position_ik_local_step_clip_scale == pytest.approx(2.5)
    assert tuned.position_inactive_enter_threshold_scale == pytest.approx(0.5)
    assert tuned.position_joint_velocity_limit_profile == "rm65_position_2x"
    metadata = real_teleop._position_ik_iteration_metadata(tuned)
    assert metadata["position_ik_orientation_weight"] == pytest.approx(0.5)
    assert metadata["position_ik_solve_max_iterations"] == 4
    assert metadata["position_ik_solve_min_iterations"] == 3
    assert metadata["position_ik_damping"] == pytest.approx(0.04)
    assert metadata["position_ik_adaptive_iterations"] is True
    assert metadata["position_inactive_enter_threshold_scale"] == pytest.approx(0.5)
    assert metadata["position_joint_velocity_limit_profile"] == "rm65_position_2x"

    with pytest.raises(SystemExit):
        _parse_real_teleop_args(
            [
                "--arm-execution-profile",
                "impedance",
                "--ik-solve-max-iterations",
                "2",
                "--position-ik-local-step-clip-scale",
                "2",
                "--position-inactive-enter-threshold-scale",
                "0.5",
                "--position-joint-velocity-limit-profile",
                "rm65_position_2x",
            ]
        )
    with pytest.raises(SystemExit):
        _parse_real_teleop_args(
            [
                "--arm-execution-profile",
                "impedance",
                "--position-ik-orientation-weight",
                "0.5",
            ]
        )


def test_position_runtime_ik_weight_and_inactive_tightening_are_scoped() -> None:
    args = _parse_real_teleop_args(
        [
            "--arm-execution-profile",
            "position",
            "--position-ik-orientation-weight",
            "0.5",
            "--position-inactive-enter-threshold-scale",
            "0.5",
        ]
    )
    tuning = real_teleop._position_ik_tuning_for_args(
        args,
        load_ik_tuning_config(),
        arm_control_mode="position",
    )
    objective = tuning.objective_weights()
    assert objective.single_arm_position_weight == pytest.approx(1.0)
    assert objective.single_arm_orientation_weight == pytest.approx(0.5)
    assert objective.dual_arm_position_cost == pytest.approx(1.0)
    assert objective.dual_arm_orientation_cost == pytest.approx(0.5)

    intent = real_teleop._position_tracker_motion_intent_config(
        args,
        arm_control_mode="position",
    )
    assert intent.enter_linear_speed_m_s == pytest.approx(0.015)
    assert intent.enter_angular_speed_rad_s == pytest.approx(0.09)
    assert intent.enter_translation_span_m == pytest.approx(0.00125)
    assert intent.enter_rotation_span_rad == pytest.approx(0.0125)
    assert intent.enter_dwell_sec == pytest.approx(0.30)
    assert intent.exit_linear_speed_m_s == pytest.approx(0.060)

    base_tuning = load_ik_tuning_config()
    assert real_teleop._position_ik_tuning_for_args(
        args,
        base_tuning,
        arm_control_mode="impedance",
    ) is base_tuning
    impedance_intent = real_teleop._position_tracker_motion_intent_config(
        args,
        arm_control_mode="impedance",
    )
    assert impedance_intent.enter_linear_speed_m_s == pytest.approx(0.030)
    assert impedance_intent.enter_translation_span_m == pytest.approx(0.0025)


def test_position_singularity_preferences_have_atomic_rollback_switches() -> None:
    enabled = _parse_real_teleop_args(
        [
            "--position-control-profile",
            "rm65_bimanual_position_v1",
            "--arm-control-mode",
            "position",
        ]
    )
    assert enabled.position_prefer_non_elbow_singular_branch is True
    assert enabled.position_prefer_non_shoulder_singular_branch is True
    assert enabled.position_ik_candidate_certification_reserve_ms == pytest.approx(3.5)
    assert enabled.position_singularity_candidate_certificate is True
    enabled_metadata = real_teleop._position_ik_iteration_metadata(enabled)
    assert enabled_metadata["position_prefer_non_elbow_singular_branch"] is True
    assert enabled_metadata["position_prefer_non_shoulder_singular_branch"] is True

    rollback = _parse_real_teleop_args(
        [
            "--position-control-profile",
            "rm65_bimanual_position_v1",
            "--arm-control-mode",
            "position",
            "--no-position-prefer-non-elbow-singular-branch",
            "--no-position-prefer-non-shoulder-singular-branch",
            "--position-ik-candidate-certification-reserve-ms",
            "0",
            "--no-position-singularity-candidate-certificate",
        ]
    )
    assert rollback.position_prefer_non_elbow_singular_branch is False
    assert rollback.position_prefer_non_shoulder_singular_branch is False
    assert rollback.position_ik_candidate_certification_reserve_ms == pytest.approx(0.0)
    assert rollback.position_singularity_candidate_certificate is False
    rollback_metadata = real_teleop._position_ik_iteration_metadata(rollback)
    assert rollback_metadata["position_prefer_non_elbow_singular_branch"] is False
    assert rollback_metadata["position_prefer_non_shoulder_singular_branch"] is False

    legacy = _parse_real_teleop_args(["--arm-execution-profile", "position"])
    assert legacy.position_prefer_non_shoulder_singular_branch is False
    assert legacy.position_ik_candidate_certification_reserve_ms == pytest.approx(0.0)
    assert legacy.position_singularity_candidate_certificate is False


@pytest.mark.parametrize("value", ["0", "-0.5", "1.01", "nan"])
def test_position_inactive_tightening_rejects_invalid_scale(value: str) -> None:
    with pytest.raises(SystemExit):
        _parse_real_teleop_args(
            [
                "--arm-execution-profile",
                "position",
                "--position-inactive-enter-threshold-scale",
                value,
            ]
        )


@pytest.mark.parametrize("value", ["0", "-0.5", "nan", "inf"])
def test_position_ik_orientation_weight_rejects_invalid_value(value: str) -> None:
    with pytest.raises(SystemExit):
        _parse_real_teleop_args(
            [
                "--arm-execution-profile",
                "position",
                "--position-ik-orientation-weight",
                value,
            ]
        )


def test_legacy_common_position_profile_keeps_15_45_150_clocks() -> None:
    rates = POSITION_45_150_PROFILE.rates

    assert rates.sample_hz == 15.0
    assert rates.reference_hz == 45.0
    assert rates.ik_hz == 45.0
    assert rates.command_hz == 150.0
    assert rates.ordering_warnings == ()
    assert rates.to_metadata()["independent_clocks"] is True


def test_common_position_rates_reject_non_positive_values_but_report_ordering() -> None:
    with pytest.raises(ValueError, match="sample_hz"):
        PositionControlRates(sample_hz=0.0)

    rates = PositionControlRates(
        sample_hz=30.0,
        reference_hz=20.0,
        ik_hz=20.0,
        command_hz=10.0,
    )
    assert rates.ordering_warnings == (
        "reference_hz_below_sample_hz",
        "command_hz_below_reference_hz",
    )


def test_shared_impedance_profile_keeps_15hz_outer_and_40hz_backend_by_default() -> None:
    args = _execution_args(arm_execution_profile="impedance_40hz")

    resolved = resolve_real_arm_execution_config(
        args,
        raw_argv=("--arm-execution-profile", "impedance_40hz"),
    )

    assert resolved.requested_profile == "impedance_40hz"
    assert resolved.resolved_profile == "impedance_40hz"
    assert resolved.mode == "impedance"
    assert resolved.sample_hz == 15.0
    assert resolved.outer_control_hz == 15.0
    assert resolved.dual_arm_ik_hz == 15.0
    assert resolved.arm_control_hz == 40.0
    assert resolved.arm_canfd_follow is False
    assert resolved.position_background_streaming_enabled is False


def test_singularity_qpos_smoothing_is_position_only() -> None:
    assert (
        output_qpos_smoothing_singularity_alpha(
            "impedance",
            slow_alpha=1.0,
            fast_alpha=0.45,
        )
        is None
    )
    assert output_qpos_smoothing_singularity_alpha(
        "position",
        slow_alpha=0.14,
        fast_alpha=0.45,
    ) == pytest.approx(0.65)
    assert output_qpos_smoothing_singularity_alpha(
        "position",
        slow_alpha=0.8,
        fast_alpha=0.9,
    ) == pytest.approx(0.9)


def test_legacy_position_profile_selects_mode_and_resolves_15_45_150() -> None:
    args = _execution_args(arm_execution_profile="position_45_150hz")

    resolved = resolve_real_arm_execution_config(
        args,
        raw_argv=("--arm-execution-profile", "position_45_150hz"),
    )

    assert resolved.requested_profile == "position_45_150hz"
    assert resolved.resolved_profile == "position_45_150hz"
    assert resolved.mode == "position"
    assert resolved.sample_hz == 15.0
    assert resolved.outer_control_hz == 45.0
    assert resolved.position_reference_hz == 45.0
    assert resolved.dual_arm_ik_hz == 45.0
    assert resolved.arm_control_hz == 150.0
    assert resolved.position_background_streaming_hz == 150.0
    assert resolved.arm_canfd_follow is True
    assert resolved.position_background_streaming_enabled is True
    assert resolved.arm_canfd_trajectory_mode == 0
    assert resolved.arm_canfd_radio == 0


def test_default_position_profile_selects_mode_and_resolves_15_60_120() -> None:
    args = _execution_args(arm_execution_profile="position")

    resolved = resolve_real_arm_execution_config(
        args,
        raw_argv=("--arm-execution-profile", "position"),
    )

    assert resolved.requested_profile == "position"
    assert resolved.resolved_profile == "position"
    assert resolved.mode == "position"
    assert resolved.sample_hz == 15.0
    assert resolved.outer_control_hz == 60.0
    assert resolved.position_reference_hz == 60.0
    assert resolved.dual_arm_ik_hz == 60.0
    assert resolved.arm_control_hz == 120.0
    assert resolved.position_background_streaming_hz == 120.0
    assert resolved.arm_canfd_follow is True
    assert resolved.position_background_streaming_enabled is True


def test_unnamed_position_mode_uses_default_15_60_120_profile() -> None:
    args = _execution_args(arm_control_mode="position")

    resolved = resolve_real_arm_execution_config(args)

    assert resolved.requested_profile is None
    assert resolved.resolved_profile == "position"
    assert resolved.sample_hz == 15.0
    assert resolved.position_reference_hz == 60.0
    assert resolved.dual_arm_ik_hz == 60.0
    assert resolved.arm_control_hz == 120.0
    assert resolved.position_post_ik_qpos_smoothing_alpha == pytest.approx(0.9)


def test_position_profile_allows_stage_overrides_and_marks_legacy_stream_precedence() -> None:
    args = _execution_args(
        arm_execution_profile="position_45_150hz",
        hz=12.0,
        position_reference_hz=48.0,
        dual_arm_ik_hz=48.0,
        arm_control_hz=160.0,
        arm_canfd_follow=True,
        arm_canfd_follow_stream_hz=140.0,
        arm_follow_qpos_alpha=1.0,
    )
    raw_argv = (
        "--arm-execution-profile",
        "position_45_150hz",
        "--hz",
        "12",
        "--position-reference-hz",
        "48",
        "--dual-arm-ik-hz",
        "48",
        "--arm-control-hz",
        "160",
        "--arm-canfd-follow",
        "--arm-canfd-follow-stream-hz",
        "140",
        "--arm-follow-qpos-alpha",
        "1.0",
    )

    resolved = resolve_real_arm_execution_config(args, raw_argv=raw_argv)

    assert resolved.resolved_profile == "position_custom"
    assert resolved.sample_hz == 12.0
    assert resolved.position_reference_hz == 48.0
    assert resolved.dual_arm_ik_hz == 48.0
    assert resolved.arm_control_hz == 160.0
    assert resolved.position_background_streaming_hz == 140.0
    assert resolved.arm_follow_qpos_alpha == 1.0
    assert "arm_canfd_follow_stream_hz" in resolved.explicit_overrides
    assert any(
        "overrides --arm-control-hz" in warning
        for warning in resolved.compatibility_warnings
    )


def test_position_profile_ik_rate_follows_custom_reference_when_not_overridden() -> None:
    args = _execution_args(
        arm_execution_profile="position_45_150hz",
        position_reference_hz=50.0,
    )

    resolved = resolve_real_arm_execution_config(
        args,
        raw_argv=(
            "--arm-execution-profile",
            "position_45_150hz",
            "--position-reference-hz",
            "50",
        ),
    )

    assert resolved.position_reference_hz == 50.0
    assert resolved.outer_control_hz == 50.0
    assert resolved.dual_arm_ik_hz == 50.0
    assert resolved.resolved_profile == "position_custom"


def test_mismatched_ik_nominal_rate_is_audited_as_not_an_independent_clock() -> None:
    args = _execution_args(
        arm_execution_profile="position_45_150hz",
        dual_arm_ik_hz=30.0,
    )
    resolved = resolve_real_arm_execution_config(
        args,
        raw_argv=(
            "--arm-execution-profile",
            "position_45_150hz",
            "--dual-arm-ik-hz",
            "30",
        ),
    )

    assert any(
        "actual IK calls" in warning for warning in resolved.compatibility_warnings
    )
    assert (
        resolved.to_metadata()["rate_semantics"]["dual_arm_ik_hz"]
        == "solver_nominal_budget_not_independent_scheduler"
    )


def test_named_profile_rejects_explicit_conflicting_mode() -> None:
    args = _execution_args(
        arm_execution_profile="position_45_150hz",
        arm_control_mode="impedance",
    )

    with pytest.raises(ValueError, match="requires --arm-control-mode position"):
        resolve_real_arm_execution_config(
            args,
            raw_argv=(
                "--arm-execution-profile",
                "position_45_150hz",
                "--arm-control-mode",
                "impedance",
            ),
        )


def test_control_debug_contract_is_observational_only() -> None:
    args = SimpleNamespace(
        control_debug="full",
        control_debug_root="teleop_records/custom_control_debug",
        control_debug_queue_size=256,
        control_debug_flush_every=12,
        safe_debug=False,
        disable_hold=False,
        strict_degree_limit=False,
    )

    resolved = resolve_control_debug_config(args)

    assert resolved == ControlDebugConfig(
        level="full",
        root="teleop_records/custom_control_debug",
        queue_size=256,
        flush_every=12,
    )
    assert resolved.enabled is True
    assert resolved.record_summary is True
    assert resolved.record_full_trace is True
    assert resolved.to_metadata()["observational_only"] is True
    assert args.safe_debug is False
    assert args.disable_hold is False
    assert args.strict_degree_limit is False


def test_control_debug_runtime_metadata_distinguishes_request_attachment_and_error() -> None:
    args = SimpleNamespace(
        control_debug_config=ControlDebugConfig(level="full"),
        control_debug_runtime_metadata={},
    )

    requested = real_teleop._merge_control_debug_runtime_metadata(
        args,
        session_sidecar_requested=True,
        session_sidecar_attached=False,
        session_sidecar_error=None,
    )
    attached = real_teleop._merge_control_debug_runtime_metadata(
        args,
        session_sidecar_attached=True,
        session_sidecar_output_dir="teleop_records/control_debug/session",
        session_sidecar_attached_wall_time=123.0,
        session_sidecar_attached_monotonic_time=45.0,
    )

    assert requested["session_sidecar_attached"] is False
    assert attached["session_sidecar_requested"] is True
    assert attached["session_sidecar_attached"] is True
    assert attached["session_sidecar_error"] is None
    assert attached["session_sidecar_output_dir"].endswith("/session")
    assert attached["session_sidecar_attached_wall_time"] == 123.0
    assert attached["session_sidecar_attached_monotonic_time"] == 45.0
    assert requested is not attached
    metadata = real_teleop._debug_timing_metadata(args)["control_debug"]
    assert metadata["session_sidecar_requested"] is True
    assert metadata["session_sidecar_attached"] is True


def test_summary_recorder_embeds_sanitized_execution_context(tmp_path: Path) -> None:
    args = _parse_real_teleop_args(
        [
            "--arm-execution-profile",
            "position_45_150hz",
            "--control-debug",
            "summary",
        ]
    )
    recorder = real_teleop._new_position_debug_recorder(
        args,
        output_dir=tmp_path,
        thread_name="unit-position-summary",
        components=None,
    )
    recorder.close()

    summary = json.loads((tmp_path / "position_summary.json").read_text())
    context = summary["context"]
    assert context["arm_execution"]["resolved_profile"] == "position_45_150hz"
    assert context["arm_execution"]["rates"]["position_reference_hz"] == 45.0
    assert context["effective_transport"]["position_background_streaming_hz"] == 150.0
    assert "left_arm_ip" not in json.dumps(context)


def test_real_entry_profile_and_debug_parse_without_overwriting_legacy_raw_defaults() -> None:
    args = _parse_real_teleop_args(
        [
            "--arm-execution-profile",
            "position_45_150hz",
            "--control-debug",
            "full",
            "--control-debug-root",
            "teleop_records/profile_debug",
            "--control-debug-queue-size",
            "512",
            "--control-debug-flush-every",
            "20",
        ]
    )

    assert args.arm_control_mode == "position"
    assert args.arm_control_hz is None
    assert args.dual_arm_ik_hz is None
    assert args.arm_canfd_follow is None
    assert real_teleop._effective_outer_control_hz(args) == 45.0
    assert real_teleop._effective_dual_arm_ik_hz(args) == 45.0
    assert real_teleop._effective_arm_control_hz(args) == 150.0
    assert real_teleop._effective_arm_canfd_follow(args) is True
    assert args.control_debug_config.level == "full"
    assert args.control_debug_config.root == "teleop_records/profile_debug"
    assert args.safe_debug is False
    assert args.disable_hold is False
    assert args.strict_degree_limit is False
    metadata = real_teleop._debug_timing_metadata(args)
    assert metadata["control_debug"]["observational_only"] is True
    assert metadata["control_debug"]["recording_requested"] is True


def test_no_debug_disk_write_overrides_all_standalone_debug_writers() -> None:
    args = _parse_real_teleop_args(
        [
            "--no-debug-disk-write",
            "--control-debug",
            "full",
            "--debug-action",
            "--debug-timing",
            "--record",
            "--debug-trajectory-recording",
            "--episode-root",
            "teleop_records/debug_episodes",
        ]
    )

    assert args.no_debug_disk_write is True
    assert args.control_debug_config.level == "off"
    assert args.debug_action is False
    assert args.debug_timing_log is False
    assert args.debug_timing_auto_log is False
    assert args.no_record is True
    assert args.no_episode_recording is True
    assert args.debug_trajectory_recording is False
    modules = real_teleop._debug_timing_modules(args)
    assert modules.console is True
    assert modules.light_log is False
    assert modules.rm65_backend is True
    assert modules.ik_profile is True
    assert modules.watchdog is True
    metadata = real_teleop._debug_timing_metadata(args)
    assert metadata["debug_disk_write_disabled"] is True
    assert metadata["control_debug"]["recording_requested"] is False


def test_real_entry_legacy_position_overrides_remain_compatible() -> None:
    args = _parse_real_teleop_args(
        [
            "--arm-control-mode",
            "position",
            "--arm-control-hz",
            "80",
            "--no-arm-canfd-follow",
        ]
    )

    assert args.arm_execution_profile is None
    assert args.arm_control_hz == 80.0
    assert args.arm_canfd_follow is False
    assert args.arm_execution_config.resolved_profile == "position_custom"
    assert real_teleop._effective_arm_control_hz(args) == 80.0
    assert real_teleop._effective_arm_canfd_follow(args) is False
    assert (
        real_teleop._canfd_follow_runtime_config(args)[
            "position_background_streaming_enabled"
        ]
        is False
    )

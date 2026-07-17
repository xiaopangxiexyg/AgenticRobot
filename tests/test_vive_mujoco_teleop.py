from __future__ import annotations

from sim.contracts import TimingSpec
from teleop_runtime.debug.run_vive_mujoco_teleop import _position_timing, parse_args


def test_vive_mujoco_debug_defaults_are_explicit_real_aligned_rates() -> None:
    timing = _position_timing(TimingSpec(physics_hz=1000.0, camera_hz=30.0))
    assert timing.sample_hz == 15.0
    assert timing.reference_hz == 60.0
    assert timing.ik_hz == 60.0
    assert timing.command_hz == 120.0
    assert timing.physics_hz == 1000.0
    assert timing.parameter_source == "vive_mujoco_position_15_60_120_debug"


def test_vive_mujoco_debug_position_defaults_match_current_profile() -> None:
    args = parse_args(
        [
            "--left-tool-profile",
            "left.json",
            "--right-tool-profile",
            "right.json",
        ]
    )
    assert args.position_ik_orientation_cost == 1.0
    assert args.position_ik_solve_max_iterations == 4
    assert args.position_joint_velocity_limit_profile == "rm65_position_8x"
    assert args.position_ik_local_step_clip_scale == 8.0
    assert args.position_post_ik_qpos_smoothing_alpha == 0.14
    assert args.position_post_ik_qpos_smoothing_fast_alpha == 0.45
    assert args.position_inactive_enter_threshold_scale == 0.5
    assert args.position_target_velocity_filter_tau_sec == 0.03
    assert args.position_max_command_step_deg == 0.675

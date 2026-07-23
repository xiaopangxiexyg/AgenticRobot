from __future__ import annotations

from sim.contracts import TimingSpec
from robot_motion.rm65 import (
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_BIMANUAL_POSITION_V1_QREF_PROFILE,
)
from teleop_runtime.debug import run_vive_mujoco_teleop as entry
from teleop_runtime.debug.run_vive_mujoco_teleop import (
    _position_timing,
    _resolve_q6_ports,
    parse_args,
)


def test_vive_mujoco_debug_defaults_are_explicit_real_aligned_rates() -> None:
    timing = _position_timing(TimingSpec(physics_hz=1000.0, camera_hz=30.0))
    assert timing.sample_hz == 15.0
    assert timing.reference_hz == 60.0
    assert timing.ik_hz == 60.0
    assert timing.command_hz == 120.0
    assert timing.physics_hz == 1000.0
    assert timing.parameter_source == RM65_BIMANUAL_POSITION_V1_NAME


def test_vive_mujoco_debug_position_defaults_match_current_profile() -> None:
    args = parse_args(
        [
            "--left-tool-profile",
            "left.json",
            "--right-tool-profile",
            "right.json",
        ]
    )
    assert args.position_ik_orientation_cost == 0.35
    assert args.position_ik_solve_max_iterations == 4
    assert args.position_joint_velocity_limit_profile == RM65_BIMANUAL_POSITION_V1_QREF_PROFILE
    assert args.position_ik_local_step_clip_scale == 8.0
    assert args.position_post_ik_qpos_smoothing_alpha == 0.14
    assert args.position_post_ik_qpos_smoothing_fast_alpha == 0.45
    assert args.position_inactive_enter_threshold_scale == 0.5
    assert args.position_target_velocity_filter_tau_sec == 0.03
    assert args.position_max_command_step_deg == 0.675
    assert not hasattr(args, "position_sender")
    assert args.gripper_source_dim == 2
    assert args.gripper_command_alpha == 1.0
    assert args.gripper_input_min == 0.1
    assert args.gripper_input_max == 0.9
    assert args.viewer is False


def test_explicit_q6_ports_take_precedence_without_scanning(monkeypatch) -> None:
    args = parse_args(
        [
            "--left-tool-profile",
            "left.json",
            "--right-tool-profile",
            "right.json",
            "--left-hand-port",
            "/dev/q6-left",
            "--right-hand-port",
            "/dev/q6-right",
            "--q6-config-required",
        ]
    )

    def _unexpected_loader(**_kwargs):
        raise AssertionError("the assignment loader must not override explicit ports")

    monkeypatch.setattr(entry, "load_latest_q6_ports", _unexpected_loader)
    assert _resolve_q6_ports(args) == {
        "left": "/dev/q6-left",
        "right": "/dev/q6-right",
    }


def test_q6_assignment_only_fills_missing_explicit_side(monkeypatch) -> None:
    args = parse_args(
        [
            "--left-tool-profile",
            "left.json",
            "--right-tool-profile",
            "right.json",
            "--left-hand-port",
            "/dev/q6-left-explicit",
        ]
    )
    calls = []

    def _load_ports(**kwargs):
        calls.append(kwargs)
        return {"right": "/dev/q6-right-config"}

    monkeypatch.setattr(entry, "load_latest_q6_ports", _load_ports)
    assert _resolve_q6_ports(args) == {
        "left": "/dev/q6-left-explicit",
        "right": "/dev/q6-right-config",
    }
    assert calls == [
        {
            "active_sides": ("right",),
            "config": "latest",
            "allow_missing": True,
        }
    ]

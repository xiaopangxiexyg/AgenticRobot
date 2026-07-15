from __future__ import annotations

from types import SimpleNamespace

from teleop_runtime.defaults import (
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from teleop_runtime.real_teleop.control_profile import (
    arm_control_mode,
    canfd_follow_runtime_config,
    effective_arm_canfd_follow,
    effective_arm_control_hz,
    effective_arm_follow_qpos_alpha,
    effective_dual_arm_ik_hz,
    effective_dual_arm_temporary_safety,
    effective_outer_control_hz,
    effective_position_post_ik_qpos_smoothing_alpha,
    effective_position_reference_hz,
    position_retargeter_stability_switches,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "arm_execution_config": None,
        "arm_control_mode": "impedance",
        "hz": DEFAULT_TELEOP_SAMPLE_HZ,
        "position_reference_hz": DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        "dual_arm_ik_hz": None,
        "arm_control_hz": None,
        "arm_canfd_follow": None,
        "arm_canfd_follow_stream_hz": None,
        "arm_canfd_follow_warning_ratio": None,
        "arm_canfd_trajectory_mode": None,
        "arm_canfd_radio": None,
        "arm_follow_qpos_alpha": None,
        "position_post_ik_qpos_smoothing_alpha": None,
        "dual_arm_temporary_safety": True,
        "safe_debug": False,
        "disable_hold": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_legacy_profile_helpers_are_direct_aliases_of_the_extracted_component() -> None:
    assert legacy_entrypoint._arm_control_mode_arg is arm_control_mode
    assert legacy_entrypoint._effective_position_reference_hz is effective_position_reference_hz
    assert legacy_entrypoint._effective_outer_control_hz is effective_outer_control_hz
    assert legacy_entrypoint._effective_dual_arm_ik_hz is effective_dual_arm_ik_hz
    assert legacy_entrypoint._effective_arm_control_hz is effective_arm_control_hz
    assert legacy_entrypoint._effective_arm_canfd_follow is effective_arm_canfd_follow
    assert (
        legacy_entrypoint._effective_position_post_ik_qpos_smoothing_alpha
        is effective_position_post_ik_qpos_smoothing_alpha
    )
    assert (
        legacy_entrypoint._effective_arm_follow_qpos_alpha
        is effective_arm_follow_qpos_alpha
    )
    assert (
        legacy_entrypoint._effective_dual_arm_temporary_safety
        is effective_dual_arm_temporary_safety
    )
    assert (
        legacy_entrypoint._position_retargeter_stability_switches
        is position_retargeter_stability_switches
    )
    assert legacy_entrypoint._canfd_follow_runtime_config is canfd_follow_runtime_config


def test_position_profile_defaults_and_canfd_runtime_are_resolved_without_runtime_objects() -> None:
    args = _args(arm_control_mode="position")

    assert arm_control_mode(args) == "position"
    assert effective_position_reference_hz(args) == DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert effective_outer_control_hz(args) == DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert effective_dual_arm_ik_hz(args) == DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert effective_arm_control_hz(args) == DEFAULT_RM65_POSITION_ARM_CONTROL_HZ
    assert effective_arm_canfd_follow(args) is True
    assert canfd_follow_runtime_config(args) == {
        "position_background_streaming_enabled": True,
        "position_background_streaming_hz": float(DEFAULT_RM65_POSITION_ARM_CONTROL_HZ),
        "canfd_follow_warning_ratio": 0.9,
        "canfd_trajectory_mode": 0,
        "canfd_radio": 0,
    }
    assert effective_position_post_ik_qpos_smoothing_alpha(args) == 0.9
    assert effective_arm_follow_qpos_alpha(args) == 0.1


def test_disabled_canfd_stream_keeps_the_non_passthrough_safety_path() -> None:
    args = _args(arm_control_mode="position", arm_canfd_follow_stream_hz=0.0)

    runtime = canfd_follow_runtime_config(args)
    assert runtime["position_background_streaming_enabled"] is False
    assert effective_position_post_ik_qpos_smoothing_alpha(args) == 1.0
    assert effective_arm_follow_qpos_alpha(args) == 1.0
    assert effective_dual_arm_temporary_safety(
        args,
        position_canfd_passthrough=False,
    ) == {
        "requested": True,
        "enabled": True,
        "disabled_reason": None,
    }


def test_impedance_uses_pre_ik_coupled_safety_instead_of_post_ik_output_limiter() -> None:
    args = _args(arm_control_mode="impedance")

    assert effective_dual_arm_temporary_safety(
        args,
        position_canfd_passthrough=False,
    ) == {
        "requested": True,
        "enabled": False,
        "disabled_reason": "impedance_uses_pre_ik_coupled_safety_bounds",
    }


def test_retargeter_stability_switches_only_allow_explicit_safe_debug_hold_disable() -> None:
    args = _args(safe_debug=True, disable_hold=True)

    assert position_retargeter_stability_switches(
        args,
        position_canfd_passthrough=False,
    )["inactive_side_hold_enabled"] is True
    assert position_retargeter_stability_switches(
        args,
        position_canfd_passthrough=True,
    ) == {
        "target_pose_smoothing_enabled": True,
        "local_step_candidate_selection_enabled": True,
        "inactive_side_hold_enabled": False,
        "inactive_side_hold_execution_cap_enabled": False,
        "hold_disabled": True,
    }

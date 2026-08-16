from __future__ import annotations

import argparse

import pytest

from teleop_runtime.backends.grippers import DMTacGripperBackendConfig
from teleop_runtime.deploy import hand_backend_config
from teleop_runtime.deploy import hand_backend_runtime
from teleop_runtime.deploy import run_policy


_MIGRATED_NAMES = (
    "_apply_gripper_policy_to_dmtac_config",
    "_dmtac_tactile_enabled",
    "_hand_backend_mode",
    "_hand_backend_mode_for_side",
    "_load_gripper_task_profile",
    "_profile_value",
    "_resolved_gripper_policy",
)


def test_run_policy_preserves_migrated_hand_config_symbols() -> None:
    for name in _MIGRATED_NAMES:
        assert getattr(run_policy, name) is getattr(hand_backend_config, name)
    assert run_policy.GRIPPER_POLICY_DEFAULTS is (
        hand_backend_config.GRIPPER_POLICY_DEFAULTS
    )


def test_hand_runtime_reexports_config_owned_backend_modes() -> None:
    assert hand_backend_runtime.HAND_BACKEND_CHOICES is (
        hand_backend_config.HAND_BACKEND_CHOICES
    )
    assert hand_backend_runtime.HAND_BACKEND_INSPIRE == "inspire"
    assert hand_backend_runtime.HAND_BACKEND_DMTAC_GRIPPER == "dmtac-gripper"


def test_hand_backend_mode_resolution_preserves_default_override_and_errors() -> None:
    args = argparse.Namespace(
        hand_backend="inspire",
        left_hand_backend=None,
        right_hand_backend="dmtac-gripper",
    )

    assert hand_backend_config._hand_backend_mode(args) == "inspire"
    assert hand_backend_config._hand_backend_mode_for_side(args, "LEFT") == "inspire"
    assert hand_backend_config._hand_backend_mode_for_side(args, "right") == (
        "dmtac-gripper"
    )
    with pytest.raises(ValueError, match="Unsupported hand backend side 'center'"):
        hand_backend_config._hand_backend_mode_for_side(args, "center")

    args.hand_backend = "unsupported"
    with pytest.raises(ValueError, match="Unsupported hand backend 'unsupported'"):
        hand_backend_config._hand_backend_mode(args)


def test_resolved_gripper_policy_preserves_defaults_and_copying_cache() -> None:
    args = argparse.Namespace(gripper_task_profile="")

    first = hand_backend_config._resolved_gripper_policy(args)
    first["speed"] = 0.9
    second = hand_backend_config._resolved_gripper_policy(args)

    assert second == hand_backend_config.GRIPPER_POLICY_DEFAULTS
    assert second is not args._resolved_gripper_policy
    assert second["speed"] == 0.2


def test_apply_gripper_policy_preserves_shadow_no_write_flags() -> None:
    args = argparse.Namespace(
        dmtac_tactile=False,
        dmtac_tactile_timeout_ms=125,
        dmtac_tactile_wait_for_new=True,
        dmtac_tactile_fields=None,
        dmtac_tactile_compute_profile=None,
        dmtac_tactile_sequential_read=True,
        rtc_real_shadow=True,
    )
    policy = {
        **hand_backend_config.GRIPPER_POLICY_DEFAULTS,
        "speed": 0.3,
        "torque": 0.4,
        "tactile_fields": ("raw",),
    }
    config = DMTacGripperBackendConfig(side="left")

    updated = hand_backend_config._apply_gripper_policy_to_dmtac_config(
        config,
        args=args,
        policy=policy,
    )

    assert updated is config
    assert updated.speed == 0.3
    assert updated.torque == 0.4
    assert updated.tactile_timeout_ms == 125
    assert updated.tactile_wait_for_new is True
    assert updated.tactile_parallel_read is False
    assert updated.connect_tactile is False
    assert updated.tactile_ids == ()
    assert updated.observation_only is True
    assert updated.send_commands is False
    assert updated.clear_fault_on_start is False
    assert updated.open_on_start is False
    assert updated.tactile_reset_on_start is False
    assert updated.tactile_warmup_frames == 0
    assert updated.enable_telemetry is True

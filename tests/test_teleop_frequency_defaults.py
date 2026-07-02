from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

from teleop_runtime.collection import run_collect
from teleop_runtime.defaults import DEFAULT_RM65_ARM_CONTROL_HZ, DEFAULT_TELEOP_SAMPLE_HZ
from teleop_runtime.deploy import run_policy


def _parse_collect_args(argv: list[str]):
    with patch.object(sys, "argv", ["prog", *argv]):
        return run_collect.parse_args()


def _parse_deploy_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    return parser.parse_args(argv)


def test_collect_defaults_to_15hz_for_bimanual() -> None:
    args = _parse_collect_args(["--active-sides", "both", "--no-camera", "--no-video"])

    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.nominal_fps == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.arm_control_hz == DEFAULT_RM65_ARM_CONTROL_HZ


def test_collect_defaults_to_15hz_for_single_arm() -> None:
    args = _parse_collect_args(["--active-sides", "right", "--no-camera", "--no-video"])

    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.nominal_fps == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.arm_control_hz == DEFAULT_RM65_ARM_CONTROL_HZ


def test_collect_nominal_fps_follows_explicit_hz_when_omitted() -> None:
    args = _parse_collect_args(["--active-sides", "right", "--no-camera", "--no-video", "--hz", "12"])

    assert args.hz == 12.0
    assert args.nominal_fps == 12.0


def test_deploy_default_fallback_action_hz_is_15hz_for_bimanual() -> None:
    args = _parse_deploy_args(["--active-sides", "both"])
    active_sides = run_policy.validate_args(args)
    hz, info = run_policy.resolve_effective_action_hz(args, policy=None)

    assert active_sides == ("left", "right")
    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.arm_control_hz == DEFAULT_RM65_ARM_CONTROL_HZ
    assert hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert info["source"] == "cli.hz"


def test_deploy_default_fallback_action_hz_is_15hz_for_single_arm() -> None:
    args = _parse_deploy_args(["--active-sides", "right"])
    active_sides = run_policy.validate_args(args)
    hz, info = run_policy.resolve_effective_action_hz(args, policy=None)

    assert active_sides == ("right",)
    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.arm_control_hz == DEFAULT_RM65_ARM_CONTROL_HZ
    assert hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert info["source"] == "cli.hz"


def test_deploy_dry_run_arm_backend_uses_40hz_control_default() -> None:
    args = _parse_deploy_args(["--active-sides", "right"])
    active_sides = run_policy.validate_args(args)

    backends = run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert backends["right"].config.control_dt == 1.0 / DEFAULT_RM65_ARM_CONTROL_HZ

from __future__ import annotations

import argparse

import pytest

from teleop_runtime.deploy.tactile_cli_validation import validate_tactile_cli_args


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "hand_tactile_regions": None,
        "send_actions": False,
        "arms_enabled": True,
        "left_arm_ip": "10.0.0.11",
        "right_arm_ip": "10.0.0.12",
        "left_hand_ip": "10.0.0.21",
        "right_hand_ip": "10.0.0.22",
        "left_hand_backend": "inspire",
        "right_hand_backend": "inspire",
        "visualize_tactile": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _mode(args: argparse.Namespace, side: str) -> str:
    return str(getattr(args, f"{side}_hand_backend"))


def _validate(
    args: argparse.Namespace,
    *,
    active_sides: tuple[str, ...] = ("left", "right"),
    end_effector_sides: tuple[str, ...] = (),
    hands_enabled: bool = False,
) -> None:
    validate_tactile_cli_args(
        args,
        active_sides=active_sides,
        end_effector_sides=end_effector_sides,
        hands_enabled=hands_enabled,
        hand_backend_mode_for_side=_mode,
    )


def test_tactile_cli_defaults_are_admitted_without_io() -> None:
    _validate(_args())


def test_send_actions_requires_arm_ip_for_each_active_side() -> None:
    with pytest.raises(
        ValueError,
        match="--send-actions requires --left-arm-ip for active side 'left'",
    ):
        _validate(_args(send_actions=True, left_arm_ip=""), active_sides=("left",))


def test_send_actions_requires_inspire_hand_ip() -> None:
    with pytest.raises(
        ValueError, match="--send-actions with hands requires --left-hand-ip"
    ):
        _validate(
            _args(send_actions=True, left_hand_ip=""),
            active_sides=("left",),
            end_effector_sides=("left",),
            hands_enabled=True,
        )


def test_send_actions_allows_disconnected_dmtac_hand() -> None:
    _validate(
        _args(send_actions=True, left_hand_ip="", left_hand_backend="dmtac-gripper"),
        active_sides=("left",),
        end_effector_sides=("left",),
        hands_enabled=True,
    )


def test_visualize_tactile_rejects_dmtac_gripper() -> None:
    with pytest.raises(
        ValueError,
        match="--visualize-tactile currently supports InspireTac UV tactile",
    ):
        _validate(
            _args(
                visualize_tactile=True,
                left_hand_backend="dmtac-gripper",
            ),
            end_effector_sides=("left",),
            hands_enabled=True,
        )


def test_visualize_tactile_requires_hands() -> None:
    with pytest.raises(ValueError, match="--visualize-tactile requires hands to be enabled"):
        _validate(_args(visualize_tactile=True))


def test_visualize_tactile_requires_inspire_hand_ip() -> None:
    with pytest.raises(
        ValueError, match="--visualize-tactile requires --left-hand-ip"
    ):
        _validate(
            _args(visualize_tactile=True, left_hand_ip=""),
            end_effector_sides=("left",),
            hands_enabled=True,
        )


def test_visualize_tactile_accepts_inspire_hand_with_ip() -> None:
    _validate(
        _args(visualize_tactile=True),
        end_effector_sides=("left",),
        hands_enabled=True,
    )

from __future__ import annotations

import argparse

import pytest

from teleop_runtime.collection import collection_cli_validation, run_collect


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "hand_backend": "inspire",
        "left_hand_backend": None,
        "right_hand_backend": None,
        "left_arm_ip": "10.0.0.11",
        "right_arm_ip": "10.0.0.12",
        "left_hand_ip": "10.0.0.21",
        "right_hand_ip": "10.0.0.22",
        "left_hand_port": "22001",
        "right_hand_port": "22002",
        "left_gripper_port": "",
        "right_gripper_port": "",
        "visualize_tactile": False,
        "with_tactile": True,
        "hand_tactile_regions": "all",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_collection_validation_helpers_remain_compatibility_exports() -> None:
    assert run_collect._active_side_value is collection_cli_validation._active_side_value
    assert run_collect._hand_backend_mode is collection_cli_validation.hand_backend_mode
    assert (
        run_collect._hand_backend_mode_for_side
        is collection_cli_validation.hand_backend_mode_for_side
    )
    assert (
        run_collect._parse_hand_tactile_regions
        is collection_cli_validation.parse_hand_tactile_regions
    )
    assert (
        run_collect._assert_side_config_consistency
        is collection_cli_validation.validate_side_config
    )
    assert (
        run_collect._assert_tactile_visualization_config
        is collection_cli_validation.validate_tactile_visualization_config
    )


def test_side_validation_requires_each_active_arm_endpoint() -> None:
    with pytest.raises(
        ValueError,
        match="--left-arm-ip is required when active-sides includes 'left'",
    ):
        collection_cli_validation.validate_side_config(
            _args(left_arm_ip=""),
            ("left",),
        )


def test_side_validation_rejects_configuration_for_inactive_side() -> None:
    with pytest.raises(
        ValueError,
        match=r"Inactive side 'right' received configuration for \['hand_ip'\]",
    ):
        collection_cli_validation.validate_side_config(
            _args(
                right_arm_ip="",
                right_hand_ip="10.0.0.22",
                right_hand_port="",
            ),
            ("left",),
        )


def test_tactile_visualization_requires_tactile_polling() -> None:
    with pytest.raises(
        ValueError,
        match="--visualize-tactile requires tactile polling",
    ):
        collection_cli_validation.validate_tactile_visualization_config(
            _args(visualize_tactile=True, with_tactile=False),
            ("left",),
        )


def test_tactile_visualization_rejects_dmtac_backend() -> None:
    with pytest.raises(
        ValueError,
        match="--visualize-tactile currently supports InspireTac UV tactile only",
    ):
        collection_cli_validation.validate_tactile_visualization_config(
            _args(visualize_tactile=True, left_hand_backend="dmtac-gripper"),
            ("left",),
        )


def test_tactile_visualization_requires_regions_and_active_hand_ips() -> None:
    with pytest.raises(ValueError, match="requires non-empty --hand-tactile-regions"):
        collection_cli_validation.validate_tactile_visualization_config(
            _args(visualize_tactile=True, hand_tactile_regions=None),
            ("left",),
        )

    with pytest.raises(ValueError, match=r"missing=\['right'\]"):
        collection_cli_validation.validate_tactile_visualization_config(
            _args(visualize_tactile=True, right_hand_ip=""),
            ("left", "right"),
        )


def test_tactile_visualization_accepts_inspire_backend_with_regions_and_ips() -> None:
    collection_cli_validation.validate_tactile_visualization_config(
        _args(visualize_tactile=True),
        ("left",),
    )


def test_hand_backend_mode_preserves_default_override_and_errors() -> None:
    args = _args(right_hand_backend="dmtac-gripper")
    assert collection_cli_validation.hand_backend_mode(args) == "inspire"
    assert collection_cli_validation.hand_backend_mode_for_side(args, "LEFT") == "inspire"
    assert (
        collection_cli_validation.hand_backend_mode_for_side(args, "right")
        == "dmtac-gripper"
    )
    with pytest.raises(ValueError, match="Unsupported hand backend side 'center'"):
        collection_cli_validation.hand_backend_mode_for_side(args, "center")

    args.hand_backend = "unsupported"
    with pytest.raises(ValueError, match="Unsupported hand backend 'unsupported'"):
        collection_cli_validation.hand_backend_mode(args)

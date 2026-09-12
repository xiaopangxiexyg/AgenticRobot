from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleop_runtime.real_teleop.side_mapping import (
    build_vive_tracker_configs_for_args,
    opposite_side,
    resolve_active_arm_ip_by_side,
    resolve_logical_side_assignments,
    tracker_legacy_index_for_source_side,
    tracker_name_for_source_side,
    tracker_role_for_source_side,
    tracker_source_side_for_logical_side,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint


def test_legacy_side_mapping_helpers_are_direct_aliases() -> None:
    assert legacy_entrypoint._resolve_logical_side_assignments is resolve_logical_side_assignments
    assert legacy_entrypoint._opposite_side is opposite_side
    assert (
        legacy_entrypoint._tracker_source_side_for_logical_side
        is tracker_source_side_for_logical_side
    )
    assert legacy_entrypoint._tracker_name_for_source_side is tracker_name_for_source_side
    assert legacy_entrypoint._tracker_role_for_source_side is tracker_role_for_source_side
    assert (
        legacy_entrypoint._tracker_legacy_index_for_source_side
        is tracker_legacy_index_for_source_side
    )
    assert legacy_entrypoint._build_vive_tracker_configs_for_args is build_vive_tracker_configs_for_args


def test_extracted_side_mapping_keeps_arm_and_operator_swaps_independent() -> None:
    args = SimpleNamespace(
        swap_arm_sides=True,
        swap_operator_sides=True,
        left_arm_ip="10.0.0.11",
        right_arm_ip="10.0.0.12",
        left_tracker_name="tracker-left",
        right_tracker_name="tracker-right",
        aux_tracker_name="tracker-aux",
    )

    assert resolve_logical_side_assignments(args) == {
        "left_arm_ip": ("right", "10.0.0.11"),
        "right_arm_ip": ("left", "10.0.0.12"),
    }
    configs = build_vive_tracker_configs_for_args(args, ("left", "right"))
    assert [
        (config.name, config.side, config.role, config.channel_name, config.legacy_index)
        for config in configs
    ] == [
        ("tracker-right", "left", "right_wrist", "left_wrist_pose", 1),
        ("tracker-left", "right", "left_wrist", "right_wrist_pose", 0),
        ("tracker-aux", None, "auxiliary", "auxiliary_tracker_pose", 2),
    ]
    assert configs[-1].critical is False


def test_active_arm_endpoint_resolution_rejects_shared_physical_arm() -> None:
    args = SimpleNamespace(
        swap_arm_sides=False,
        left_arm_ip="192.168.1.20",
        right_arm_ip="192.168.1.20",
    )

    with pytest.raises(ValueError, match="must use distinct endpoints"):
        resolve_active_arm_ip_by_side(args, ("left", "right"))

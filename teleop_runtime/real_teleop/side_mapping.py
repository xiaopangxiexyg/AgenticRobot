"""Pure logical-side and Vive tracker mapping for the real-teleop entrypoint.

The module maps parsed command-line assignments into ``ViveTrackerConfig``
objects only.  It deliberately does not start sources or mutate live runtime
objects, so tracker transport/lifecycle remains owned by the entrypoint.
"""

from __future__ import annotations

import argparse

from ..sources.pose import ViveTrackerConfig


def resolve_logical_side_assignments(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    return {
        "left_arm_ip": ("right" if args.swap_arm_sides else "left", args.left_arm_ip),
        "right_arm_ip": ("left" if args.swap_arm_sides else "right", args.right_arm_ip),
    }


def opposite_side(side: str) -> str:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    raise ValueError(f"side must be left or right, got {side!r}")


def tracker_source_side_for_logical_side(args: argparse.Namespace, logical_side: str) -> str:
    side = str(logical_side)
    return opposite_side(side) if bool(getattr(args, "swap_operator_sides", False)) else side


def tracker_name_for_source_side(args: argparse.Namespace, source_side: str) -> str:
    if source_side == "left":
        return str(args.left_tracker_name)
    if source_side == "right":
        return str(args.right_tracker_name)
    raise ValueError(f"source_side must be left or right, got {source_side!r}")


def tracker_role_for_source_side(source_side: str) -> str:
    if source_side == "left":
        return "left_wrist"
    if source_side == "right":
        return "right_wrist"
    raise ValueError(f"source_side must be left or right, got {source_side!r}")


def tracker_legacy_index_for_source_side(source_side: str) -> int:
    if source_side == "left":
        return 0
    if source_side == "right":
        return 1
    raise ValueError(f"source_side must be left or right, got {source_side!r}")


def build_vive_tracker_configs_for_args(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> list[ViveTrackerConfig]:
    tracker_configs: list[ViveTrackerConfig] = []
    if "left" in active_sides:
        source_side = tracker_source_side_for_logical_side(args, "left")
        tracker_configs.append(
            ViveTrackerConfig(
                name=tracker_name_for_source_side(args, source_side),
                side="left",
                role=tracker_role_for_source_side(source_side),
                channel_name="left_wrist_pose",
                legacy_index=tracker_legacy_index_for_source_side(source_side),
            )
        )
    if "right" in active_sides:
        source_side = tracker_source_side_for_logical_side(args, "right")
        tracker_configs.append(
            ViveTrackerConfig(
                name=tracker_name_for_source_side(args, source_side),
                side="right",
                role=tracker_role_for_source_side(source_side),
                channel_name="right_wrist_pose",
                legacy_index=tracker_legacy_index_for_source_side(source_side),
            )
        )
    if args.aux_tracker_name:
        tracker_configs.append(
            ViveTrackerConfig(
                name=args.aux_tracker_name,
                side=None,
                role="auxiliary",
                channel_name="auxiliary_tracker_pose",
                legacy_index=2,
                critical=False,
            )
        )
    return tracker_configs


__all__ = [
    "build_vive_tracker_configs_for_args",
    "opposite_side",
    "resolve_logical_side_assignments",
    "tracker_legacy_index_for_source_side",
    "tracker_name_for_source_side",
    "tracker_role_for_source_side",
    "tracker_source_side_for_logical_side",
]

"""Pure endpoint and tactile admission checks for collection.

The headless collection entrypoint keeps these calls at its historical
validation boundary.  The helpers only inspect parsed CLI values and backend
mode selection; they do not load configuration, construct backends, connect
devices, or start collection lifecycle components.
"""

from __future__ import annotations

import argparse

from ..hand_mapping import parse_tactile_regions
from .collect_cli_parser import (
    HAND_BACKEND_CHOICES,
    HAND_BACKEND_DMTAC_GRIPPER,
    HAND_BACKEND_INSPIRE,
)


def parse_hand_tactile_regions(value: object) -> tuple[str, ...]:
    """Normalize collection tactile-region input without performing I/O."""

    return parse_tactile_regions(value)


def hand_backend_mode(args: argparse.Namespace) -> str:
    """Resolve the collection-wide hand backend from parsed arguments."""

    mode = str(getattr(args, "hand_backend", HAND_BACKEND_INSPIRE) or HAND_BACKEND_INSPIRE)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported hand backend {mode!r}; expected one of {HAND_BACKEND_CHOICES}."
        )
    return mode


def hand_backend_mode_for_side(args: argparse.Namespace, side: str) -> str:
    """Resolve one side's hand backend while preserving CLI compatibility."""

    side = str(side).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported hand backend side {side!r}.")
    override = getattr(args, f"{side}_hand_backend", None)
    mode = hand_backend_mode(args) if override in (None, "") else str(override)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(f"Unsupported {side} hand backend {mode!r}.")
    return mode


def _active_side_value(args: argparse.Namespace, side: str, suffix: str) -> str:
    return str(getattr(args, f"{side}_{suffix}", "") or "")


def validate_side_config(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> None:
    """Reject missing active arm endpoints and configuration for inactive sides."""

    for side in ("left", "right"):
        if side in active_sides:
            if not _active_side_value(args, side, "arm_ip"):
                raise ValueError(f"--{side}-arm-ip is required when active-sides includes '{side}'.")
            continue
        unexpected_values = {
            "arm_ip": _active_side_value(args, side, "arm_ip"),
            "hand_ip": _active_side_value(args, side, "hand_ip"),
            "hand_port": _active_side_value(args, side, "hand_port"),
            "gripper_port": _active_side_value(args, side, "gripper_port"),
        }
        present = [flag for flag, value in unexpected_values.items() if value]
        if present:
            raise ValueError(
                f"Inactive side '{side}' received configuration for {present}; "
                "either remove those flags or include the side in --active-sides."
            )


def validate_tactile_visualization_config(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> None:
    """Validate tactile visualization prerequisites without touching devices."""

    if not bool(args.visualize_tactile):
        return
    if not bool(args.with_tactile):
        raise ValueError(
            "--visualize-tactile requires tactile polling; remove --without-tactile."
        )
    if any(
        hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
        for side in active_sides
    ):
        raise ValueError(
            "--visualize-tactile currently supports InspireTac UV tactile only; "
            "DMTAC tactile is recorded but needs a separate visualizer."
        )
    tactile_regions = parse_hand_tactile_regions(args.hand_tactile_regions)
    if not tactile_regions:
        raise ValueError(
            "--visualize-tactile requires non-empty --hand-tactile-regions, for example "
            "'thumb_1,index_1,palm' or 'all'."
        )
    missing_ip_sides = [
        side for side in active_sides if not str(getattr(args, f"{side}_hand_ip") or "")
    ]
    if missing_ip_sides:
        raise ValueError(
            "--visualize-tactile requires real InspireTac hand IPs for active sides; "
            f"missing={missing_ip_sides}."
        )


__all__ = [
    "hand_backend_mode",
    "hand_backend_mode_for_side",
    "parse_hand_tactile_regions",
    "validate_side_config",
    "validate_tactile_visualization_config",
]

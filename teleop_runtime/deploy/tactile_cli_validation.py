"""Pure tactile and endpoint admission checks for policy deployment.

The deployment facade keeps this call at its historical validation point.  The
checks only inspect parsed CLI values and backend-mode selection; they do not
construct hand backends, connect devices, or dispatch actions.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from ..hand_mapping import parse_tactile_regions
from .hand_backend_config import HAND_BACKEND_DMTAC_GRIPPER, HAND_BACKEND_INSPIRE


def validate_tactile_cli_args(
    args: argparse.Namespace,
    *,
    active_sides: tuple[str, ...],
    end_effector_sides: tuple[str, ...],
    hands_enabled: bool,
    hand_backend_mode_for_side: Callable[[argparse.Namespace, str], str],
) -> None:
    """Validate tactile settings and endpoint requirements without I/O."""

    parse_tactile_regions(args.hand_tactile_regions)
    if bool(args.send_actions) and bool(args.arms_enabled):
        for side in active_sides:
            if not getattr(args, f"{side}_arm_ip"):
                raise ValueError(
                    f"--send-actions requires --{side}-arm-ip for active side '{side}'."
                )
    if bool(args.send_actions) and hands_enabled:
        for side in end_effector_sides:
            if (
                hand_backend_mode_for_side(args, side) == HAND_BACKEND_INSPIRE
                and not getattr(args, f"{side}_hand_ip")
            ):
                raise ValueError(
                    f"--send-actions with hands requires --{side}-hand-ip."
                )
    if bool(args.visualize_tactile):
        if any(
            hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
            for side in end_effector_sides
        ):
            raise ValueError(
                "--visualize-tactile currently supports InspireTac UV tactile, not DMTAC gripper tactile."
            )
        if not hands_enabled:
            raise ValueError("--visualize-tactile requires hands to be enabled.")
        for side in end_effector_sides:
            if not getattr(args, f"{side}_hand_ip"):
                raise ValueError(f"--visualize-tactile requires --{side}-hand-ip.")


__all__ = ["validate_tactile_cli_args"]

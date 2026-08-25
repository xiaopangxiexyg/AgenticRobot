"""Pure hand-action schema selection for the collection compatibility path.

This module maps parsed collection backend modes to the hand action contracts
that the dataset writer expects.  It does not construct a backend, connect a
device, or dispatch a command.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

from ..action import HandActionSpec
from .collect_cli_parser import HAND_BACKEND_DMTAC_GRIPPER
from .collection_cli_validation import hand_backend_mode_for_side


def resolve_collection_hand_action_specs(
    args: argparse.Namespace,
    active_sides: Sequence[str],
) -> Mapping[str, HandActionSpec] | None:
    """Return DMTAC hand action specs, preserving the legacy ``None`` case."""

    dmtac_sides = [
        str(side)
        for side in active_sides
        if hand_backend_mode_for_side(args, str(side)) == HAND_BACKEND_DMTAC_GRIPPER
    ]
    if not dmtac_sides:
        return None
    return {side: HandActionSpec.dmtac_gripper(side) for side in dmtac_sides}


__all__ = ["resolve_collection_hand_action_specs"]

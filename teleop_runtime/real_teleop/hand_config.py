"""Hand-backend selection for the real-teleop runtime.

This module owns only argument normalization and side selection.  It does not
construct, start, or communicate with any hand or gripper backend.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


HAND_BACKEND_INSPIRE = "inspire"
HAND_BACKEND_DMTAC_GRIPPER = "dmtac-gripper"
HAND_BACKEND_NONE = "none"
HAND_BACKEND_CHOICES = (
    HAND_BACKEND_INSPIRE,
    HAND_BACKEND_DMTAC_GRIPPER,
    HAND_BACKEND_NONE,
)


def hand_backend_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "hand_backend", HAND_BACKEND_INSPIRE) or HAND_BACKEND_INSPIRE)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(f"Unsupported hand backend {mode!r}; expected one of {HAND_BACKEND_CHOICES}.")
    return mode


def hand_backend_mode_for_side(args: argparse.Namespace, side: str) -> str:
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError(f"Unsupported hand backend side {side!r}.")
    override = getattr(args, f"{normalized_side}_hand_backend", None)
    mode = hand_backend_mode(args) if override in (None, "") else str(override)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported {normalized_side} hand backend {mode!r}; "
            f"expected one of {HAND_BACKEND_CHOICES}."
        )
    return mode


def dmtac_sides(args: argparse.Namespace, active_sides: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        str(side)
        for side in active_sides
        if hand_backend_mode_for_side(args, str(side)) == HAND_BACKEND_DMTAC_GRIPPER
    )


__all__ = [
    "HAND_BACKEND_CHOICES",
    "HAND_BACKEND_DMTAC_GRIPPER",
    "HAND_BACKEND_INSPIRE",
    "HAND_BACKEND_NONE",
    "dmtac_sides",
    "hand_backend_mode",
    "hand_backend_mode_for_side",
]

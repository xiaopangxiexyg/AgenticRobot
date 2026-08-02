"""Frozen logical topology of the current dual-wrist UMI capture adapter.

This is an application/source-adapter declaration, not a dataset schema. A
future topology receives a separate adapter while the existing v2.3 and v2.5-p
projections remain unchanged.
"""

from __future__ import annotations


SIDES = ("left", "right")
CAMERA_KEY_BY_SIDE = {
    "left": "left_wrist",
    "right": "right_wrist",
}
CAMERA_KEYS = tuple(CAMERA_KEY_BY_SIDE[side] for side in SIDES)
HAND_EFFECTOR_BY_SIDE = {
    "left": "left_hand",
    "right": "right_hand",
}
DEFAULT_COMMAND_CHANNEL_BY_SIDE = {
    "left": "left_angle_trigger",
    "right": "right_angle_trigger",
}


__all__ = [
    "CAMERA_KEYS",
    "CAMERA_KEY_BY_SIDE",
    "DEFAULT_COMMAND_CHANNEL_BY_SIDE",
    "HAND_EFFECTOR_BY_SIDE",
    "SIDES",
]

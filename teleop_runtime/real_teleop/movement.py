"""Movement-multiplier state shared by the real-teleop runtime and GUI."""

from __future__ import annotations

import argparse

from .cli_value_parsers import _movement_multiple_arg
from .components import RealTeleopComponents
from ..log_utils import log_event as _log


MOVEMENT_MULTIPLE_DEFAULT = 1.35
MOVEMENT_MULTIPLE_GUI_MIN = 0.05
MOVEMENT_MULTIPLE_GUI_MAX = 10.0
MOVEMENT_MULTIPLE_GUI_STEP = 0.1
MOVEMENT_MULTIPLE_GUI_EVENTS = (
    "movement_multiple_down",
    "movement_multiple_up",
    "movement_multiple_reset",
    "movement_multiple_set",
)


def resolve_movement_multiple_args(
    args: argparse.Namespace,
    *,
    raw_arg_set: set[str] | None = None,
) -> float:
    raw_arg_set = raw_arg_set or set()
    movement_multiple = getattr(args, "movement_multiple", None)
    if movement_multiple is None:
        movement_multiple = getattr(args, "translation_scale", MOVEMENT_MULTIPLE_DEFAULT)
    value = _movement_multiple_arg(str(movement_multiple))
    if "--movement-multiple" in raw_arg_set and "--translation-scale" in raw_arg_set:
        _log(
            "MovementMultiple",
            "--movement-multiple overrides legacy --translation-scale.",
        )
    setattr(args, "movement_multiple", value)
    setattr(args, "translation_scale", value)
    return value


def movement_multiple_value(args: argparse.Namespace) -> float:
    value = getattr(args, "movement_multiple", None)
    if value is None:
        value = getattr(args, "translation_scale", MOVEMENT_MULTIPLE_DEFAULT)
    return _movement_multiple_arg(str(value))


def movement_multiple_ui_state(args: argparse.Namespace) -> dict[str, object]:
    value = movement_multiple_value(args)
    return {
        "movement_multiple": value,
        "movement_multiple_min": MOVEMENT_MULTIPLE_GUI_MIN,
        "movement_multiple_max": max(MOVEMENT_MULTIPLE_GUI_MAX, value),
        "movement_multiple_step": MOVEMENT_MULTIPLE_GUI_STEP,
    }


def apply_movement_multiple(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    value: float,
) -> float:
    movement_multiple = _movement_multiple_arg(str(value))
    setattr(args, "movement_multiple", movement_multiple)
    setattr(args, "translation_scale", movement_multiple)
    for retargeter in (components.left_retargeter, components.right_retargeter):
        retargeter.config.translation_scale = movement_multiple
    if components.dual_arm_retargeter is not None:
        components.dual_arm_retargeter.config.translation_scale = movement_multiple
        components.dual_arm_retargeter.left_pose_retargeter.config.translation_scale = movement_multiple
        components.dual_arm_retargeter.right_pose_retargeter.config.translation_scale = movement_multiple
    calibration = components.session.calibration
    if calibration is not None:
        calibration.metadata["movement_multiple"] = movement_multiple
        calibration.metadata["translation_scale"] = movement_multiple
    return movement_multiple


__all__ = [
    "MOVEMENT_MULTIPLE_DEFAULT",
    "MOVEMENT_MULTIPLE_GUI_EVENTS",
    "MOVEMENT_MULTIPLE_GUI_MAX",
    "MOVEMENT_MULTIPLE_GUI_MIN",
    "MOVEMENT_MULTIPLE_GUI_STEP",
    "apply_movement_multiple",
    "movement_multiple_ui_state",
    "movement_multiple_value",
    "resolve_movement_multiple_args",
]

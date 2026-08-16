"""Value parsers owned by the real-policy deployment CLI.

The parsers in this module are pure conversions: they do not inspect runtime
state or initialize deployment resources.  Their leading-underscore names are
kept because :mod:`teleop_runtime.deploy.run_policy` historically exposed the
same helpers to programmatic callers.
"""

from __future__ import annotations

import argparse

from robot_motion.deploy.schema import ALL_SIDES

from ..cli_value_parsers import _non_negative_int, _positive_float, _positive_int
from ..dmtac_tactile_fields import normalize_dmtac_tactile_fields


def _unit_interval_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return value


def _parse_dmtac_tactile_fields(text: str) -> tuple[str, ...]:
    try:
        return normalize_dmtac_tactile_fields(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_active_sides(
    value: str | list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip().lower()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
    else:
        parts = [
            part.strip().lower()
            for part in str(value).replace(",", " ").split()
            if part.strip()
        ]
    if not parts:
        raise argparse.ArgumentTypeError(
            "active-sides must be one of: left, right, both"
        )
    if parts == ["both"]:
        return ("left", "right")
    invalid = sorted(set(parts) - set(ALL_SIDES))
    if invalid:
        raise argparse.ArgumentTypeError(
            "active-sides must be one of: left, right, both"
        )
    return tuple(side for side in ALL_SIDES if side in set(parts))


def _parse_disconnected_gripper_sides(
    value: str | list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip().lower()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
    else:
        parts = [
            part.strip().lower()
            for part in str(value).replace(",", " ").split()
            if part.strip()
        ]
    if not parts or parts == ["none"]:
        return ()
    if parts == ["both"]:
        return tuple(ALL_SIDES)
    invalid = sorted(set(parts) - set(ALL_SIDES))
    if invalid:
        raise argparse.ArgumentTypeError(
            "disconnected-gripper-sides must be none, left, right, or both"
        )
    return tuple(side for side in ALL_SIDES if side in set(parts))


def _parse_end_effector_sides(
    value: str | list[str] | tuple[str, ...],
    *,
    active_sides: tuple[str, ...],
    hands_enabled: bool,
) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip().lower()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
    else:
        parts = [
            part.strip().lower()
            for part in str(value).replace(",", " ").split()
            if part.strip()
        ]
    if not parts or parts == ["active"]:
        return tuple(active_sides) if hands_enabled else ()
    if parts == ["none"]:
        return ()
    if parts == ["both"]:
        parts = ["left", "right"]
    invalid = set(parts) - set(ALL_SIDES)
    if invalid:
        raise argparse.ArgumentTypeError(
            "end-effector-sides must be active, none, left, right, or both"
        )
    if not hands_enabled:
        raise ValueError("--end-effector-sides requires hands to be enabled.")
    requested = set(parts)
    unavailable = requested - set(active_sides)
    if unavailable:
        raise ValueError(
            "--end-effector-sides must be a subset of --active-sides, got "
            + ",".join(sorted(unavailable))
        )
    return tuple(side for side in ALL_SIDES if side in requested)


def _parse_float_tuple(text: str, *, length: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != int(length):
        raise argparse.ArgumentTypeError(
            f"{name} must contain {length} comma-separated floats"
        )
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {name}: {text}") from exc
    return values


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _parse_vector3(
    value: str | list[str] | tuple[object, ...],
) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
        text = " ".join(parts)
    else:
        text = str(value)
        parts = [
            part.strip() for part in text.replace(",", " ").split() if part.strip()
        ]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("vector3 must contain 3 floats")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid vector3: {text}") from exc
    return (values[0], values[1], values[2])


def _parse_qpos6(text: str) -> tuple[float, ...]:
    return _parse_float_tuple(text, length=6, name="qpos6")


def _parse_arm_max_angvel(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("arm max angvel must contain 1 or 6 floats")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid arm max angvel: {text}") from exc
    if len(values) == 1:
        if values[0] <= 0.0:
            raise argparse.ArgumentTypeError("arm max angvel must be positive")
        return (float(values[0]),) * 6
    if len(values) != 6 or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError(
            "arm max angvel must contain 1 or 6 positive floats"
        )
    return tuple(float(value) for value in values)


__all__ = [
    "_non_negative_float",
    "_non_negative_int",
    "_parse_active_sides",
    "_parse_arm_max_angvel",
    "_parse_disconnected_gripper_sides",
    "_parse_dmtac_tactile_fields",
    "_parse_end_effector_sides",
    "_parse_float_tuple",
    "_parse_qpos6",
    "_parse_vector3",
    "_positive_float",
    "_positive_int",
    "_unit_interval_float",
]

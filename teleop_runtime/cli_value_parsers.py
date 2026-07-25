"""Backend-neutral ``argparse`` value parsers shared by teleop frontends."""

from __future__ import annotations

import argparse


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _alpha_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be in (0, 1]")
    return value


def _parse_joint_order(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("joint order must contain 6 comma-separated indices")
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid joint order: {text}") from exc
    if sorted(order) != [0, 1, 2, 3, 4, 5]:
        raise argparse.ArgumentTypeError("joint order must be a permutation of 0,1,2,3,4,5")
    return order


def _parse_arm_max_angvel(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("arm max angvel must be a float or 6 comma-separated floats")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid arm max angvel: {text}") from exc
    if len(values) == 1:
        value = float(values[0])
        if value <= 0.0:
            raise argparse.ArgumentTypeError("arm max angvel must be positive")
        return (value,) * 6
    if len(values) != 6:
        raise argparse.ArgumentTypeError("arm max angvel must contain 1 or 6 floats")
    if any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("arm max angvel must be positive")
    return tuple(float(value) for value in values)


def _parse_active_sides(text: str) -> tuple[str, ...]:
    value = str(text).strip().lower()
    if value == "both":
        return ("left", "right")
    if value in {"left", "right"}:
        return (value,)
    raise argparse.ArgumentTypeError("active-sides must be one of: left, right, both")


def _normalize_active_sides(value: object) -> tuple[str, ...]:
    if value is None:
        return ("left", "right")
    if isinstance(value, str):
        return _parse_active_sides(value)
    sides = tuple(str(side).strip().lower() for side in value if str(side).strip())
    if sides == ("both",):
        return ("left", "right")
    invalid = set(sides) - {"left", "right"}
    if invalid:
        raise ValueError(f"Unsupported active side(s): {sorted(invalid)}")
    ordered = tuple(side for side in ("left", "right") if side in set(sides))
    if not ordered:
        raise ValueError("At least one active side is required.")
    return ordered


def _active_sides_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    return _normalize_active_sides(getattr(args, "active_sides", ("left", "right")))


__all__ = [
    "_active_sides_from_args",
    "_alpha_float",
    "_non_negative_int",
    "_normalize_active_sides",
    "_parse_active_sides",
    "_parse_arm_max_angvel",
    "_parse_joint_order",
    "_positive_float",
]

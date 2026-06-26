from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CollectionModeSpec:
    name: str
    fixed_arm_sides: tuple[str, ...] = ()
    jitter_arm_sides: tuple[str, ...] = ()
    moving_arm_sides: tuple[str, ...] = ()
    source_entrypoint: str = "teleop_runtime.collection"
    experimental: bool = False


def fixed_arm_sides(args: Any, active_sides: tuple[str, ...]) -> tuple[str, ...]:
    if not bool(getattr(args, "left_arm_fixed", False)):
        return ()
    if bool(getattr(args, "left_arm_jitter", False)):
        raise ValueError("--left-arm-fixed and --left-arm-jitter are mutually exclusive.")
    if set(active_sides) != {"left", "right"}:
        raise ValueError("--left-arm-fixed requires --active-sides both.")
    return ("left",)


def jitter_arm_sides(args: Any, active_sides: tuple[str, ...]) -> tuple[str, ...]:
    if not bool(getattr(args, "left_arm_jitter", False)):
        return ()
    if bool(getattr(args, "left_arm_fixed", False)):
        raise ValueError("--left-arm-fixed and --left-arm-jitter are mutually exclusive.")
    if set(active_sides) != {"left", "right"}:
        raise ValueError("--left-arm-jitter requires --active-sides both.")
    if bool(getattr(args, "disable_dual_arm_collision_avoidance", False)):
        raise ValueError(
            "--left-arm-jitter requires dual-arm IK; remove --disable-dual-arm-collision-avoidance."
        )
    return ("left",)


def moving_arm_sides(args: Any, active_sides: tuple[str, ...]) -> tuple[str, ...]:
    non_operator = set(fixed_arm_sides(args, active_sides)) | set(
        jitter_arm_sides(args, active_sides)
    )
    return tuple(side for side in active_sides if side not in non_operator)


def resolve_collection_mode(args: Any, active_sides: tuple[str, ...]) -> CollectionModeSpec:
    fixed_sides = fixed_arm_sides(args, active_sides)
    jitter_sides = jitter_arm_sides(args, active_sides)
    moving_sides = moving_arm_sides(args, active_sides)
    if jitter_sides:
        name = "left_arm_jitter"
    elif fixed_sides:
        name = "left_arm_fixed"
    elif set(active_sides) == {"left", "right"}:
        name = "bimanual_collect"
    else:
        name = f"{active_sides[0]}_arm_collect" if active_sides else "collect"
    return CollectionModeSpec(
        name=name,
        fixed_arm_sides=fixed_sides,
        jitter_arm_sides=jitter_sides,
        moving_arm_sides=moving_sides,
        source_entrypoint="teleop_runtime.collection",
        experimental=bool(fixed_sides or jitter_sides),
    )

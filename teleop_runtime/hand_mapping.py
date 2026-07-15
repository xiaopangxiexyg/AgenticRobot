from __future__ import annotations

from robot_motion.action.hand_mapping import (
    DEFAULT_INSPIRE_HAND_JOINT_ORDER,
    SOURCE_HAND_Q6_ORDER,
    TARGET_INSPIRE_HAND_Q6_ORDER,
)

ALL_TACTILE_REGION_NAMES = (
    "little_1",
    "little_2",
    "little_3",
    "ring_1",
    "ring_2",
    "ring_3",
    "middle_1",
    "middle_2",
    "middle_3",
    "index_1",
    "index_2",
    "index_3",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_4",
    "palm",
)


def parse_tactile_regions(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        parts = [str(item).strip() for item in value if str(item).strip()]
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if "all" in {part.lower() for part in parts}:
        return ALL_TACTILE_REGION_NAMES
    return tuple(parts)


__all__ = [
    "ALL_TACTILE_REGION_NAMES",
    "DEFAULT_INSPIRE_HAND_JOINT_ORDER",
    "SOURCE_HAND_Q6_ORDER",
    "TARGET_INSPIRE_HAND_Q6_ORDER",
    "parse_tactile_regions",
]

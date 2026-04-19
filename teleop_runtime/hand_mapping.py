from __future__ import annotations

SOURCE_HAND_Q6_ORDER = (
    "thumb_yaw",
    "thumb_flex",
    "index",
    "middle",
    "ring",
    "pinky",
)

# InspireTac q6 expects the four fingers first, then thumb yaw/flex.
DEFAULT_INSPIRE_HAND_JOINT_ORDER = (5, 4, 3, 2, 0, 1)

TARGET_INSPIRE_HAND_Q6_ORDER = tuple(
    SOURCE_HAND_Q6_ORDER[index] for index in DEFAULT_INSPIRE_HAND_JOINT_ORDER
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

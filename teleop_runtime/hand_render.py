from __future__ import annotations

from typing import Sequence

import numpy as np

LEFT_HAND_RENDER_JOINT_NAMES = (
    "left_thumb_1_joint",
    "left_thumb_2_joint",
    "left_thumb_3_joint",
    "left_thumb_4_joint",
    "left_index_1_joint",
    "left_index_2_joint",
    "left_middle_1_joint",
    "left_middle_2_joint",
    "left_ring_1_joint",
    "left_ring_2_joint",
    "left_little_1_joint",
    "left_little_2_joint",
)
RIGHT_HAND_RENDER_JOINT_NAMES = (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
    "right_index_1_joint",
    "right_index_2_joint",
    "right_middle_1_joint",
    "right_middle_2_joint",
    "right_ring_1_joint",
    "right_ring_2_joint",
    "right_little_1_joint",
    "right_little_2_joint",
)

OPEN_LEFT_HAND_RENDER_QPOS = np.zeros(len(LEFT_HAND_RENDER_JOINT_NAMES), dtype=float)
OPEN_RIGHT_HAND_RENDER_QPOS = np.zeros(len(RIGHT_HAND_RENDER_JOINT_NAMES), dtype=float)

_INSPIRE_PRIMARY_LIMITS = np.asarray([1.15, 0.55, 1.60, 1.60, 1.60, 1.60], dtype=float)
_INSPIRE_RENDER_LIMITS = np.asarray(
    [1.15, 0.55, 0.55 * 0.60, 0.55 * 0.80, 1.60, 1.60 * 1.05, 1.60, 1.60 * 1.05, 1.60, 1.60 * 1.05, 1.60, 1.60 * 1.18],
    dtype=float,
)


def hand_render_joint_names(side: str) -> tuple[str, ...]:
    side = str(side).strip().lower()
    if side == "left":
        return LEFT_HAND_RENDER_JOINT_NAMES
    if side == "right":
        return RIGHT_HAND_RENDER_JOINT_NAMES
    raise ValueError(f"Unsupported hand side: {side}")


def open_hand_render_qpos(side: str) -> np.ndarray:
    side = str(side).strip().lower()
    if side == "left":
        return OPEN_LEFT_HAND_RENDER_QPOS.copy()
    if side == "right":
        return OPEN_RIGHT_HAND_RENDER_QPOS.copy()
    raise ValueError(f"Unsupported hand side: {side}")


def inspire_q6_to_render_qpos(joint_pos_q6: Sequence[float] | np.ndarray) -> np.ndarray:
    q6 = np.asarray(joint_pos_q6, dtype=float).reshape(-1)
    if q6.shape != (6,):
        raise ValueError(f"InspireTac render mapping expects shape (6,), got {q6.shape}.")
    # InspireTac telemetry uses the hardware convention directly; the URDF/SAPIEN hand
    # is authored with the opposite open/close direction, so rendering must flip to 1 - x.
    q6 = 1.0 - np.clip(q6, 0.0, 1.0)
    primary = q6 * _INSPIRE_PRIMARY_LIMITS
    thumb_flex = float(primary[1])
    finger = primary[2:]
    q12 = np.asarray(
        [
            primary[0],
            thumb_flex,
            thumb_flex * 0.60,
            thumb_flex * 0.80,
            finger[0],
            finger[0] * 1.05,
            finger[1],
            finger[1] * 1.05,
            finger[2],
            finger[2] * 1.05,
            finger[3],
            finger[3] * 1.18,
        ],
        dtype=float,
    )
    return np.clip(q12, 0.0, _INSPIRE_RENDER_LIMITS)

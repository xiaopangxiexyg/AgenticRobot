from __future__ import annotations

from enum import Enum
from typing import List

import numpy as np


THUMB_P_RIGHT = [240, 239, 238, 256, 255, 254, 16, 15, 14, 32, 31, 30]
INDEX_P_RIGHT = [237, 236, 235, 253, 252, 251, 13, 12, 11, 29, 28, 27]
MIDDLE_P_RIGHT = [234, 233, 232, 250, 249, 248, 10, 9, 8, 26, 25, 24]
RING_P_RIGHT = [231, 230, 229, 247, 246, 245, 7, 6, 5, 23, 22, 21]
PINKY_P_RIGHT = [228, 227, 226, 244, 243, 242, 4, 3, 2, 20, 19, 18]
BEND_ID_RIGHT = [47, 44, 41, 38, 35]
PALM_P_RIGHT = [
    61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50,
    80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66,
    96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82,
    112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98,
    128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114,
]

THUMB_P_LEFT = [19, 18, 17, 3, 2, 1, 243, 242, 241, 227, 226, 225]
INDEX_P_LEFT = [22, 21, 20, 6, 5, 4, 246, 245, 244, 230, 229, 228]
MIDDLE_P_LEFT = [25, 24, 23, 9, 8, 7, 249, 248, 247, 233, 232, 231]
RING_P_LEFT = [28, 27, 26, 12, 11, 10, 252, 251, 250, 236, 235, 234]
PINKY_P_LEFT = [31, 30, 29, 15, 14, 13, 255, 254, 253, 239, 238, 237]
BEND_ID_LEFT = [210, 213, 216, 219, 222]
PALM_P_LEFT = [
    207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196,
    191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177,
    175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161,
    159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145,
    143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129,
]

LEFT_ORDER = THUMB_P_LEFT + INDEX_P_LEFT + MIDDLE_P_LEFT + RING_P_LEFT + PINKY_P_LEFT + BEND_ID_LEFT + PALM_P_LEFT
RIGHT_ORDER = THUMB_P_RIGHT + INDEX_P_RIGHT + MIDDLE_P_RIGHT + RING_P_RIGHT + PINKY_P_RIGHT + BEND_ID_RIGHT + PALM_P_RIGHT


class HandType(Enum):
    LEFT = "left"
    RIGHT = "right"


class HandConfig:
    def __init__(self, hand: HandType):
        self.hand = hand
        if hand == HandType.LEFT:
            self.finger_press_ids = [THUMB_P_LEFT, INDEX_P_LEFT, MIDDLE_P_LEFT, RING_P_LEFT, PINKY_P_LEFT]
            self.bend_ids = BEND_ID_LEFT
            self.palm_ids = PALM_P_LEFT
        elif hand == HandType.RIGHT:
            self.finger_press_ids = [THUMB_P_RIGHT, INDEX_P_RIGHT, MIDDLE_P_RIGHT, RING_P_RIGHT, PINKY_P_RIGHT]
            self.bend_ids = BEND_ID_RIGHT
            self.palm_ids = PALM_P_RIGHT
        else:
            raise ValueError(f"Unsupported hand type: {hand}")

        valid_ids = [sensor_id for sensor_group in self.finger_press_ids for sensor_id in sensor_group if sensor_id > 0]
        valid_ids.extend(sensor_id for sensor_id in self.bend_ids if sensor_id > 0)
        valid_ids.extend(sensor_id for sensor_id in self.palm_ids if sensor_id > 0)
        self.id2pos = {sensor_id: sensor_id - 1 for sensor_id in valid_ids}

        self.id_mats_fingers = self._create_finger_id_mats()
        self.id_mat_palm = self._create_palm_id_mat()

    def _create_finger_id_mats(self) -> List[np.ndarray]:
        mats: List[np.ndarray] = []
        for finger_index, finger_ids in enumerate(self.finger_press_ids):
            matrix = np.full((5, 3), np.nan)
            for row in range(4):
                for col in range(3):
                    sensor_id = finger_ids[row * 3 + col]
                    matrix[row, col] = sensor_id if sensor_id > 0 else np.nan
            bend_sensor = self.bend_ids[finger_index]
            matrix[4, 1] = bend_sensor if bend_sensor > 0 else np.nan
            mats.append(matrix)
        return mats

    def _create_palm_id_mat(self) -> np.ndarray:
        matrix = np.full((5, 15), np.nan)
        if self.hand == HandType.RIGHT:
            init_offset = 3
            for col in range(12):
                if col < len(self.palm_ids):
                    sensor_id = self.palm_ids[col]
                    matrix[0, col + init_offset] = sensor_id if sensor_id > 0 else np.nan
        else:
            for col in range(12):
                if col < len(self.palm_ids):
                    sensor_id = self.palm_ids[col]
                    matrix[0, col] = sensor_id if sensor_id > 0 else np.nan

        offset = 12
        for row in range(1, 5):
            for col in range(15):
                idx = offset + (row - 1) * 15 + col
                if idx < len(self.palm_ids):
                    sensor_id = self.palm_ids[idx]
                    matrix[row, col] = sensor_id if sensor_id > 0 else np.nan
        return matrix

    def extract_finger_val_mat(self, finger_idx: int, all_vals: np.ndarray) -> np.ndarray:
        id_mat = self.id_mats_fingers[finger_idx]
        val_mat = np.full_like(id_mat, np.nan, dtype=float)
        for row in range(id_mat.shape[0]):
            for col in range(id_mat.shape[1]):
                sensor_id = id_mat[row, col]
                if not np.isnan(sensor_id) and int(sensor_id) in self.id2pos:
                    val_mat[row, col] = all_vals[self.id2pos[int(sensor_id)]]
        return val_mat

    def extract_palm_val_mat(self, all_vals: np.ndarray) -> np.ndarray:
        id_mat = self.id_mat_palm
        val_mat = np.full_like(id_mat, np.nan, dtype=float)
        for row in range(id_mat.shape[0]):
            for col in range(id_mat.shape[1]):
                sensor_id = id_mat[row, col]
                if not np.isnan(sensor_id) and int(sensor_id) in self.id2pos:
                    val_mat[row, col] = all_vals[self.id2pos[int(sensor_id)]]
        return val_mat


__all__ = [
    "BEND_ID_LEFT",
    "BEND_ID_RIGHT",
    "HandConfig",
    "HandType",
    "INDEX_P_LEFT",
    "INDEX_P_RIGHT",
    "LEFT_ORDER",
    "MIDDLE_P_LEFT",
    "MIDDLE_P_RIGHT",
    "PALM_P_LEFT",
    "PALM_P_RIGHT",
    "PINKY_P_LEFT",
    "PINKY_P_RIGHT",
    "RIGHT_ORDER",
    "RING_P_LEFT",
    "RING_P_RIGHT",
    "THUMB_P_LEFT",
    "THUMB_P_RIGHT",
]

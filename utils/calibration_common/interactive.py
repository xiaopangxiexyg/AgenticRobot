"""Shared interactive mechanics for calibration tools with distinct solvers."""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from .transforms import transform_distance


class AutoRecordState:
    """Track a stable, sufficiently new robot pose before burst capture."""

    def __init__(self, args: Any):
        self.args = args
        self.reference_pose: np.ndarray | None = None
        self.stable_since = 0.0
        self.last_record_pose: np.ndarray | None = None
        self.cooldown_until = 0.0

    def update(self, robot_pose: np.ndarray, *, target_visible: bool) -> tuple[bool, str]:
        now = time.monotonic()
        if self.reference_pose is None:
            self.reference_pose = robot_pose.copy()
            self.stable_since = now
            return False, "auto: waiting for still pose"

        moved_m, moved_deg = transform_distance(self.reference_pose, robot_pose)
        if (
            moved_m > self.args.auto_still_translation_m
            or moved_deg > self.args.auto_still_rotation_deg
        ):
            self.reference_pose = robot_pose.copy()
            self.stable_since = now
            return False, f"auto: moving {moved_m * 1000.0:.1f}mm/{moved_deg:.2f}deg"

        stable_sec = max(0.0, now - self.stable_since)
        is_new, new_m, new_deg = self._is_new_pose(robot_pose)
        if now < self.cooldown_until:
            return False, "auto: cooldown"
        if not target_visible:
            return False, f"auto: still {stable_sec:.1f}s, target not visible"
        if not is_new:
            return False, (
                f"auto: still {stable_sec:.1f}s, pose already recorded "
                f"({new_m * 1000.0:.1f}mm/{new_deg:.1f}deg)"
            )
        if stable_sec < self.args.auto_stable_sec:
            return False, f"auto: still {stable_sec:.1f}/{self.args.auto_stable_sec:.1f}s"
        return True, f"auto: recording new pose ({new_m * 1000.0:.1f}mm/{new_deg:.1f}deg)"

    def mark_recorded(self, robot_pose: np.ndarray) -> None:
        now = time.monotonic()
        self.last_record_pose = robot_pose.copy()
        self.reference_pose = robot_pose.copy()
        self.stable_since = now
        self.cooldown_until = now + self.args.auto_cooldown_sec

    def mark_attempted(self) -> None:
        self.cooldown_until = time.monotonic() + self.args.auto_cooldown_sec

    def reset_last_record(self, samples: Sequence[Any]) -> None:
        self.last_record_pose = samples[-1].T_gripper_to_base.copy() if samples else None

    def _is_new_pose(self, robot_pose: np.ndarray) -> tuple[bool, float, float]:
        if self.last_record_pose is None:
            return True, math.inf, math.inf
        distance_m, distance_deg = transform_distance(self.last_record_pose, robot_pose)
        return (
            distance_m >= self.args.auto_new_pose_translation_m
            or distance_deg >= self.args.auto_new_pose_rotation_deg
        ), distance_m, distance_deg


def draw_session_overlay(
    cv2: Any,
    image: np.ndarray,
    sample_count: int,
    min_samples: int,
    status: str = "",
) -> None:
    """Draw the shared sample-count and auto-record status overlay."""

    text = f"samples {sample_count}/{min_samples}   r record   c compute   d delete   q quit"
    y0 = max(54, image.shape[0] - 34)
    box_height = 48 if status else 32
    cv2.rectangle(
        image,
        (8, y0 - 24),
        (min(image.shape[1] - 8, 920), y0 - 24 + box_height),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (18, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    if status:
        cv2.putText(
            image,
            status[:96],
            (18, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (170, 230, 170),
            1,
            cv2.LINE_AA,
        )


__all__ = ["AutoRecordState", "draw_session_overlay"]

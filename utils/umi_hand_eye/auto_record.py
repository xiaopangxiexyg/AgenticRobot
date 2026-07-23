"""State machine for recording a UMI sample after a stable pose dwell."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from utils.hand_eye.transforms import transform_distance


@dataclass(frozen=True)
class AutoRecordConfig:
    """Thresholds used by the live automatic sample trigger."""

    stable_sec: float = 0.5
    still_translation_m: float = 0.0015
    still_rotation_deg: float = 0.25
    target_still_translation_m: float = 0.003
    target_still_rotation_deg: float = 0.8
    new_pose_translation_m: float = 0.04
    new_pose_rotation_deg: float = 10.0
    require_both_pose_changes: bool = True
    cooldown_sec: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "stable_sec",
            "still_translation_m",
            "still_rotation_deg",
            "target_still_translation_m",
            "target_still_rotation_deg",
            "new_pose_translation_m",
            "new_pose_rotation_deg",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.cooldown_sec) < 0.0:
            raise ValueError("cooldown_sec must be >= 0")


class AutoRecordState:
    """Trigger once a tracker has been still at a new pose for long enough.

    ``update`` is deliberately independent of cameras and UI.  The caller
    still performs the full static burst capture after the trigger, so this
    state machine is only a candidate detector and never bypasses the burst
    motion/reprojection checks.
    """

    def __init__(self, config: AutoRecordConfig):
        self.config = config
        self.reference_pose: np.ndarray | None = None
        self.target_reference_pose: np.ndarray | None = None
        self.stable_since: float | None = None
        self.recorded_poses: list[np.ndarray] = []
        self.cooldown_until = 0.0

    @property
    def last_record_pose(self) -> np.ndarray | None:
        return self.recorded_poses[-1] if self.recorded_poses else None

    def update(
        self,
        tracker_pose: np.ndarray,
        *,
        target_visible: bool,
        target_pose: np.ndarray | None = None,
        now: float | None = None,
    ) -> tuple[bool, str]:
        current_time = time.monotonic() if now is None else float(now)
        pose = np.asarray(tracker_pose, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("tracker_pose must be a finite 4x4 matrix")
        target = None
        if target_pose is not None:
            target = np.asarray(target_pose, dtype=float)
            if target.shape != (4, 4) or not np.all(np.isfinite(target)):
                raise ValueError("target_pose must be a finite 4x4 matrix")

        if self.reference_pose is None or self.stable_since is None:
            self.reference_pose = pose.copy()
            if target_visible and target is not None:
                self.target_reference_pose = target.copy()
            self.stable_since = current_time
            return False, "auto: waiting for still pose"

        moved_m, moved_deg = transform_distance(self.reference_pose, pose)
        if (
            moved_m > float(self.config.still_translation_m)
            or moved_deg > float(self.config.still_rotation_deg)
        ):
            self.reference_pose = pose.copy()
            if target is not None:
                self.target_reference_pose = target.copy()
            self.stable_since = current_time
            return False, f"auto: moving {moved_m * 1000.0:.1f}mm/{moved_deg:.2f}deg"

        if not target_visible:
            self.target_reference_pose = None
            self.stable_since = current_time
            return False, "auto: target not visible; still timer reset"
        if target is not None:
            if self.target_reference_pose is None:
                self.target_reference_pose = target.copy()
                self.stable_since = current_time
                return False, "auto: target acquired; waiting for stable target"
            target_moved_m, target_moved_deg = transform_distance(
                self.target_reference_pose,
                target,
            )
            if (
                target_moved_m > float(self.config.target_still_translation_m)
                or target_moved_deg > float(self.config.target_still_rotation_deg)
            ):
                self.target_reference_pose = target.copy()
                self.reference_pose = pose.copy()
                self.stable_since = current_time
                return False, (
                    f"auto: target moving {target_moved_m * 1000.0:.1f}mm/"
                    f"{target_moved_deg:.2f}deg"
                )

        stable_sec = max(0.0, current_time - self.stable_since)
        is_new, new_m, new_deg = self._is_new_pose(pose)
        if current_time < self.cooldown_until:
            return False, "auto: cooldown"
        if not is_new:
            change_word = "both" if self.config.require_both_pose_changes else "either"
            return False, (
                f"auto: pose too similar ({new_m * 1000.0:.1f}mm/{new_deg:.1f}deg); "
                f"need {change_word} {float(self.config.new_pose_translation_m) * 1000.0:.0f}mm "
                f"and {float(self.config.new_pose_rotation_deg):.0f}deg"
            )
        if stable_sec < float(self.config.stable_sec):
            return False, (
                f"auto: still {stable_sec:.1f}/{float(self.config.stable_sec):.1f}s"
            )
        return True, f"auto: recording new pose ({new_m * 1000.0:.1f}mm/{new_deg:.1f}deg)"

    def mark_recorded(self, tracker_pose: np.ndarray, *, now: float | None = None) -> None:
        current_time = time.monotonic() if now is None else float(now)
        pose = np.asarray(tracker_pose, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("tracker_pose must be a finite 4x4 matrix")
        self.recorded_poses.append(pose.copy())
        self.reference_pose = pose.copy()
        self.stable_since = current_time
        self.cooldown_until = current_time + float(self.config.cooldown_sec)

    def mark_attempted(self, *, now: float | None = None) -> None:
        current_time = time.monotonic() if now is None else float(now)
        self.cooldown_until = current_time + float(self.config.cooldown_sec)
        self.stable_since = current_time

    def reset_last_record(self, poses: list[np.ndarray]) -> None:
        self.recorded_poses = [np.asarray(pose, dtype=float).copy() for pose in poses]

    def _is_new_pose(self, tracker_pose: np.ndarray) -> tuple[bool, float, float]:
        if not self.recorded_poses:
            return True, math.inf, math.inf
        distances = [transform_distance(recorded, tracker_pose) for recorded in self.recorded_poses]
        translation_limit = float(self.config.new_pose_translation_m)
        rotation_limit = float(self.config.new_pose_rotation_deg)
        if self.config.require_both_pose_changes:
            invalid = [
                value
                for value in distances
                if value[0] < translation_limit or value[1] < rotation_limit
            ]
        else:
            invalid = [
                value
                for value in distances
                if value[0] < translation_limit and value[1] < rotation_limit
            ]
        candidates = invalid or distances
        nearest = min(
            candidates,
            key=lambda value: (
                float(value[0]) / translation_limit
                + float(value[1]) / rotation_limit
            ),
        )
        return not invalid, float(nearest[0]), float(nearest[1])


__all__ = ["AutoRecordConfig", "AutoRecordState"]

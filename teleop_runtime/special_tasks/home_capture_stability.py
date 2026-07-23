from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from robot_motion.pose import pose_delta_metrics


# Runtime Home is an operator calibration boundary, not an ordinary tracker
# sample.  Require a short continuous still interval and then keep the accepted
# window compact enough that slow drift cannot silently become a new anchor.
HOME_CAPTURE_STILL_TRANSLATION_SPEED_M_S = 0.015
HOME_CAPTURE_STILL_ROTATION_SPEED_DEG_S = 2.5
HOME_CAPTURE_STILL_DWELL_SEC = 0.25
HOME_CAPTURE_MAX_SAMPLE_GAP_SEC = 0.5
HOME_CAPTURE_MAX_POSITION_SPREAD_M = 0.003
HOME_CAPTURE_MAX_ROTATION_SPREAD_DEG = 0.75


@dataclass(frozen=True)
class HomeCaptureSampleDecision:
    accepted: bool
    reset_samples: bool = False
    reason: str = ""


@dataclass
class HomeCaptureStabilityState:
    last_pose: np.ndarray | None = None
    last_timestamp: float | None = None
    stable_since: float | None = None
    reason: str = "waiting_for_first_tracker_sample"

    def observe(
        self,
        pose: np.ndarray,
        *,
        source_timestamp: float,
        first_accepted_pose: np.ndarray | None,
    ) -> HomeCaptureSampleDecision:
        candidate = np.asarray(pose, dtype=float)
        timestamp = float(source_timestamp)
        if candidate.shape != (4, 4) or not np.all(np.isfinite(candidate)):
            raise ValueError("tracker Home sample must be a finite 4x4 pose")
        if not np.isfinite(timestamp):
            raise ValueError("tracker Home sample timestamp must be finite")

        previous_pose = self.last_pose
        previous_timestamp = self.last_timestamp
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            self.reason = "waiting_for_fresh_tracker_sample"
            return HomeCaptureSampleDecision(False, reason=self.reason)

        self.last_pose = candidate.copy()
        self.last_timestamp = timestamp
        if previous_pose is None or previous_timestamp is None:
            self.stable_since = timestamp
            self.reason = "tracker_stillness_dwell_in_progress"
            return HomeCaptureSampleDecision(False, reason=self.reason)

        dt_sec = timestamp - previous_timestamp
        if dt_sec > HOME_CAPTURE_MAX_SAMPLE_GAP_SEC:
            self.stable_since = timestamp
            self.reason = "tracker_sample_gap_reset"
            return HomeCaptureSampleDecision(
                False,
                reset_samples=first_accepted_pose is not None,
                reason=self.reason,
            )

        moved_m, moved_deg = pose_delta_metrics(previous_pose, candidate)
        translation_speed = float(moved_m / dt_sec)
        rotation_speed = float(moved_deg / dt_sec)
        if (
            translation_speed > HOME_CAPTURE_STILL_TRANSLATION_SPEED_M_S
            or rotation_speed > HOME_CAPTURE_STILL_ROTATION_SPEED_DEG_S
        ):
            self.stable_since = timestamp
            self.reason = "tracker_motion_above_stillness_threshold"
            return HomeCaptureSampleDecision(
                False,
                reset_samples=first_accepted_pose is not None,
                reason=self.reason,
            )

        if self.stable_since is None:
            self.stable_since = previous_timestamp
        if timestamp - self.stable_since < HOME_CAPTURE_STILL_DWELL_SEC:
            self.reason = "tracker_stillness_dwell_in_progress"
            return HomeCaptureSampleDecision(False, reason=self.reason)

        if first_accepted_pose is not None:
            spread_m, spread_deg = pose_delta_metrics(first_accepted_pose, candidate)
            if (
                spread_m > HOME_CAPTURE_MAX_POSITION_SPREAD_M
                or spread_deg > HOME_CAPTURE_MAX_ROTATION_SPREAD_DEG
            ):
                self.stable_since = timestamp
                self.reason = "tracker_capture_window_spread_reset"
                return HomeCaptureSampleDecision(
                    False,
                    reset_samples=True,
                    reason=self.reason,
                )

        self.reason = "stable_tracker_sample_accepted"
        return HomeCaptureSampleDecision(True, reason=self.reason)


__all__ = [
    "HomeCaptureSampleDecision",
    "HomeCaptureStabilityState",
]

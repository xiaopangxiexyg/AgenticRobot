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
HOME_CAPTURE_SPEED_EVALUATION_MIN_SEC = 0.05
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
    stable_anchor_pose: np.ndarray | None = None
    reason: str = "waiting_for_first_tracker_sample"

    def _restart_dwell(
        self,
        pose: np.ndarray,
        timestamp: float,
        *,
        reason: str,
        reset_samples: bool,
    ) -> HomeCaptureSampleDecision:
        self.stable_since = float(timestamp)
        self.stable_anchor_pose = np.asarray(pose, dtype=float).copy()
        self.reason = str(reason)
        return HomeCaptureSampleDecision(
            False,
            reset_samples=bool(reset_samples),
            reason=self.reason,
        )

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
            self.stable_anchor_pose = candidate.copy()
            self.reason = "tracker_stillness_dwell_in_progress"
            return HomeCaptureSampleDecision(False, reason=self.reason)

        dt_sec = timestamp - previous_timestamp
        if dt_sec > HOME_CAPTURE_MAX_SAMPLE_GAP_SEC:
            return self._restart_dwell(
                candidate,
                timestamp,
                reason="tracker_sample_gap_reset",
                reset_samples=first_accepted_pose is not None,
            )

        if self.stable_since is None or self.stable_anchor_pose is None:
            self.stable_since = previous_timestamp
            self.stable_anchor_pose = previous_pose.copy()
        dwell_sec = max(0.0, timestamp - float(self.stable_since))
        moved_m, moved_deg = pose_delta_metrics(
            self.stable_anchor_pose,
            candidate,
        )

        # VIVE can publish hundreds of samples per second.  Dividing normal
        # sub-millimetre quantization noise by one packet interval makes an
        # otherwise still tracker appear to move quickly forever.  Evaluate
        # speed over a short accumulated window instead, while the absolute
        # capture spread remains bounded on every sample.
        spread_exceeded = bool(
            moved_m > HOME_CAPTURE_MAX_POSITION_SPREAD_M
            or moved_deg > HOME_CAPTURE_MAX_ROTATION_SPREAD_DEG
        )
        speed_window_ready = dwell_sec >= HOME_CAPTURE_SPEED_EVALUATION_MIN_SEC
        translation_speed = (
            float(moved_m / dwell_sec) if speed_window_ready and dwell_sec > 0.0 else 0.0
        )
        rotation_speed = (
            float(moved_deg / dwell_sec) if speed_window_ready and dwell_sec > 0.0 else 0.0
        )
        if (
            spread_exceeded
            or (
                speed_window_ready
                and (
                    translation_speed > HOME_CAPTURE_STILL_TRANSLATION_SPEED_M_S
                    or rotation_speed > HOME_CAPTURE_STILL_ROTATION_SPEED_DEG_S
                )
            )
        ):
            return self._restart_dwell(
                candidate,
                timestamp,
                reason="tracker_motion_above_stillness_threshold",
                reset_samples=first_accepted_pose is not None,
            )

        if dwell_sec < HOME_CAPTURE_STILL_DWELL_SEC:
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
    "HOME_CAPTURE_SPEED_EVALUATION_MIN_SEC",
]

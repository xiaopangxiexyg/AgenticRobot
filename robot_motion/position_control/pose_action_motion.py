from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.pose import as_pose_matrix


POSE_ACTION_MOTION_INTENT_SCHEMA = "robot_motion.pose_action_motion_intent/v1"


@dataclass(frozen=True)
class PoseActionMotionIntentConfig:
    """§2J motion-intent thresholds for 15 Hz pose-action intervals.

    The enter thresholds are the established position-control 0.5 scale
    (0.015 m/s, 0.09 rad/s and matching span thresholds).  Exit thresholds
    intentionally retain the existing hysteresis used by the mature VIVE
    position path; this class does not import or mutate that real-teleop path.
    """

    enter_linear_speed_m_s: float = 0.015
    enter_angular_speed_rad_s: float = 0.090
    enter_translation_span_m: float = 0.00125
    enter_rotation_span_rad: float = 0.0125
    enter_dwell_sec: float = 0.30
    exit_linear_speed_m_s: float = 0.060
    exit_angular_speed_rad_s: float = 0.40
    exit_translation_m: float = 0.004
    exit_rotation_rad: float = float(np.deg2rad(2.0))
    exit_debounce_sec: float = 0.050

    def __post_init__(self) -> None:
        for name in (
            "enter_linear_speed_m_s",
            "enter_angular_speed_rad_s",
            "enter_translation_span_m",
            "enter_rotation_span_rad",
            "enter_dwell_sec",
            "exit_linear_speed_m_s",
            "exit_angular_speed_rad_s",
            "exit_translation_m",
            "exit_rotation_rad",
            "exit_debounce_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.exit_linear_speed_m_s <= self.enter_linear_speed_m_s:
            raise ValueError("exit linear speed must exceed enter linear speed")
        if self.exit_angular_speed_rad_s <= self.enter_angular_speed_rad_s:
            raise ValueError("exit angular speed must exceed enter angular speed")

    def to_metadata(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in (
                "enter_linear_speed_m_s",
                "enter_angular_speed_rad_s",
                "enter_translation_span_m",
                "enter_rotation_span_rad",
                "enter_dwell_sec",
                "exit_linear_speed_m_s",
                "exit_angular_speed_rad_s",
                "exit_translation_m",
                "exit_rotation_rad",
                "exit_debounce_sec",
            )
        }


def _rotation_angle_rad(reference: np.ndarray, target: np.ndarray) -> float:
    return float(
        Rotation.from_matrix(target[:3, :3] @ reference[:3, :3].T).magnitude()
    )


class PoseActionMotionIntentStateMachine:
    """Classify pose-action motion using source-interval speed evidence.

    Unlike the historical target-delta fallback, this consumes the velocity
    calculated from adjacent 15 Hz pose actions.  Calling it at 60 Hz therefore
    repeats the action interval's physical speed instead of shrinking the
    evidence by the interpolation ratio.
    """

    def __init__(
        self,
        config: PoseActionMotionIntentConfig | None = None,
    ) -> None:
        self.config = config or PoseActionMotionIntentConfig()
        self.reset("initialized")

    def reset(self, reason: str = "reset") -> None:
        self._state = "UNKNOWN"
        self._quiet_since: float | None = None
        self._exit_since: float | None = None
        self._last_timestamp: float | None = None
        self._quiet_anchor_pose: np.ndarray | None = None
        self._inactive_anchor_pose: np.ndarray | None = None
        self._transition_reason = str(reason)
        self._evidence: dict[str, Any] = {}

    def update(
        self,
        *,
        timestamp: float,
        pose: np.ndarray,
        interval_velocity: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        sample_time = float(timestamp)
        target_pose = as_pose_matrix(np.asarray(pose, dtype=float))
        speed = dict(interval_velocity or {})
        quality_valid = bool(speed.get("quality_valid", False))
        try:
            linear_speed = float(speed.get("linear_speed_p90_m_s"))
            angular_speed = float(speed.get("angular_speed_p90_rad_s"))
            translation_span = float(speed.get("translation_span_m"))
            rotation_span = float(speed.get("rotation_span_rad"))
        except (TypeError, ValueError):
            quality_valid = False
            linear_speed = angular_speed = translation_span = rotation_span = 0.0
        quality_valid = bool(
            quality_valid
            and math.isfinite(sample_time)
            and all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    linear_speed,
                    angular_speed,
                    translation_span,
                    rotation_span,
                )
            )
        )
        if self._last_timestamp is not None and sample_time < self._last_timestamp:
            self.reset("sample_time_rewind")
        self._last_timestamp = sample_time
        if not quality_valid:
            self.reset("pose_action_interval_velocity_invalid")
            return self._metadata(sample_time, speed)

        quiet = bool(
            linear_speed <= self.config.enter_linear_speed_m_s
            and angular_speed <= self.config.enter_angular_speed_rad_s
            and translation_span <= self.config.enter_translation_span_m
            and rotation_span <= self.config.enter_rotation_span_rad
        )
        self._evidence = {
            "quality_valid": True,
            "quiet": quiet,
            "linear_speed_p90_m_s": linear_speed,
            "angular_speed_p90_rad_s": angular_speed,
            "translation_span_m": translation_span,
            "rotation_span_rad": rotation_span,
            "speed_source": "adjacent_pose_action_interval",
        }

        quiet_anchor_translation = 0.0
        quiet_anchor_rotation = 0.0
        if self._quiet_anchor_pose is not None:
            quiet_anchor_translation = float(
                np.linalg.norm(
                    target_pose[:3, 3] - self._quiet_anchor_pose[:3, 3]
                )
            )
            quiet_anchor_rotation = _rotation_angle_rad(
                self._quiet_anchor_pose,
                target_pose,
            )
        quiet_anchor_stable = bool(
            quiet_anchor_translation <= self.config.enter_translation_span_m
            and quiet_anchor_rotation <= self.config.enter_rotation_span_rad
        )
        self._evidence.update(
            {
                "quiet_anchor_translation_m": quiet_anchor_translation,
                "quiet_anchor_rotation_rad": quiet_anchor_rotation,
                "quiet_anchor_stable": quiet_anchor_stable,
            }
        )

        if self._state == "UNKNOWN":
            self._state = "QUIETING" if quiet else "ACTIVE"
            self._quiet_since = sample_time if quiet else None
            self._quiet_anchor_pose = target_pose.copy() if quiet else None
            self._transition_reason = "quiet_evidence" if quiet else "motion_evidence"
        elif self._state == "ACTIVE":
            if quiet:
                self._state = "QUIETING"
                self._quiet_since = sample_time
                self._quiet_anchor_pose = target_pose.copy()
                self._transition_reason = "quiet_evidence"
        elif self._state == "QUIETING":
            if not quiet:
                self._state = "ACTIVE"
                self._quiet_since = None
                self._quiet_anchor_pose = None
                self._transition_reason = "quiet_evidence_lost"
            elif not quiet_anchor_stable:
                self._state = "ACTIVE"
                self._quiet_since = None
                self._quiet_anchor_pose = None
                self._transition_reason = "quiet_dwell_motion"
            elif (
                self._quiet_since is not None
                and sample_time - self._quiet_since + 1e-9
                >= self.config.enter_dwell_sec
            ):
                self._state = "INACTIVE"
                self._inactive_anchor_pose = target_pose.copy()
                self._quiet_anchor_pose = None
                self._exit_since = None
                self._transition_reason = "enter_dwell_complete"
        else:
            assert self._state == "INACTIVE"
            anchor_translation = 0.0
            anchor_rotation = 0.0
            if self._inactive_anchor_pose is not None:
                anchor_translation = float(
                    np.linalg.norm(
                        target_pose[:3, 3] - self._inactive_anchor_pose[:3, 3]
                    )
                )
                anchor_rotation = _rotation_angle_rad(
                    self._inactive_anchor_pose,
                    target_pose,
                )
            speed_exit = bool(
                linear_speed >= self.config.exit_linear_speed_m_s
                or angular_speed >= self.config.exit_angular_speed_rad_s
            )
            anchor_exit = bool(
                anchor_translation >= self.config.exit_translation_m
                or anchor_rotation >= self.config.exit_rotation_rad
            )
            self._evidence.update(
                {
                    "speed_exit": speed_exit,
                    "anchor_exit": anchor_exit,
                    "anchor_translation_m": anchor_translation,
                    "anchor_rotation_rad": anchor_rotation,
                }
            )
            if speed_exit or anchor_exit:
                if self._exit_since is None:
                    self._exit_since = sample_time
                if (
                    sample_time - self._exit_since + 1e-9
                    >= self.config.exit_debounce_sec
                ):
                    self._state = "ACTIVE"
                    self._quiet_since = None
                    self._quiet_anchor_pose = None
                    self._inactive_anchor_pose = None
                    self._exit_since = None
                    self._transition_reason = "exit_evidence_debounced"
            else:
                self._exit_since = None

        return self._metadata(sample_time, speed)

    def _metadata(
        self,
        sample_time: float,
        interval_velocity: Mapping[str, Any],
    ) -> dict[str, Any]:
        quiet_dwell = (
            0.0
            if self._quiet_since is None
            else max(0.0, sample_time - self._quiet_since)
        )
        exit_dwell = (
            0.0
            if self._exit_since is None
            else max(0.0, sample_time - self._exit_since)
        )
        return {
            "schema": POSE_ACTION_MOTION_INTENT_SCHEMA,
            "source": "pose_action_interval_velocity_state_machine",
            "state": self._state,
            "inactive": self._state == "INACTIVE",
            "hold_weight": 1.0 if self._state == "INACTIVE" else 0.0,
            "quiet_dwell_sec": quiet_dwell,
            "exit_dwell_sec": exit_dwell,
            "transition_reason": self._transition_reason,
            "evidence": dict(self._evidence),
            "motion_window": dict(interval_velocity),
            "config": self.config.to_metadata(),
        }


__all__ = [
    "POSE_ACTION_MOTION_INTENT_SCHEMA",
    "PoseActionMotionIntentConfig",
    "PoseActionMotionIntentStateMachine",
]

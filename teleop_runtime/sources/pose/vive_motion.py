from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _rotation_angle_rad(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=float).reshape(3, 3).T @ np.asarray(
        right, dtype=float
    ).reshape(3, 3)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def _finite_vector3(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(vector)):
        return None
    return vector


def summarize_tracker_motion_window(
    history: Sequence[dict[str, Any]],
    *,
    window_sec: float = 0.1,
    minimum_span_ratio: float = 0.8,
    minimum_pose_valid_ratio: float = 0.9,
    minimum_velocity_valid_ratio: float = 0.8,
    max_linear_speed_m_s: float = 5.0,
    max_angular_speed_rad_s: float = 20.0,
) -> dict[str, Any]:
    """Summarize a VIVE history window using elapsed time, not frame count."""

    requested_window_sec = max(0.0, float(window_sec))
    samples = list(history)
    if samples:
        newest_time = float(samples[-1]["history_time"])
        cutoff = newest_time - requested_window_sec
        samples = [
            sample
            for sample in samples
            if float(sample.get("history_time", float("-inf"))) >= cutoff
        ]
    else:
        newest_time = None

    valid_pose_samples: list[tuple[float, np.ndarray]] = []
    linear_speeds: list[float] = []
    angular_speeds: list[float] = []
    velocity_valid_count = 0
    for sample in samples:
        sample_time = float(sample.get("history_time", 0.0))
        pose = np.asarray(sample.get("pose"), dtype=float)
        if bool(sample.get("valid", False)) and pose.shape == (4, 4) and np.all(
            np.isfinite(pose)
        ):
            valid_pose_samples.append((sample_time, pose))

        linear_velocity = _finite_vector3(sample.get("linear_velocity_m_s"))
        angular_velocity = _finite_vector3(sample.get("angular_velocity_rad_s"))
        if (
            bool(sample.get("velocity_valid", False))
            and linear_velocity is not None
            and angular_velocity is not None
        ):
            linear_speed = float(np.linalg.norm(linear_velocity))
            angular_speed = float(np.linalg.norm(angular_velocity))
            if (
                linear_speed <= float(max_linear_speed_m_s)
                and angular_speed <= float(max_angular_speed_rad_s)
            ):
                velocity_valid_count += 1
                linear_speeds.append(linear_speed)
                angular_speeds.append(angular_speed)

    sample_count = len(samples)
    pose_valid_ratio = (
        float(len(valid_pose_samples)) / float(sample_count) if sample_count else 0.0
    )
    velocity_valid_ratio = (
        float(velocity_valid_count) / float(sample_count) if sample_count else 0.0
    )
    window_span_sec = (
        max(0.0, float(samples[-1]["history_time"]) - float(samples[0]["history_time"]))
        if len(samples) >= 2
        else 0.0
    )

    translation_span_m: float | None = None
    rotation_span_rad: float | None = None
    if valid_pose_samples:
        positions = np.stack([pose[:3, 3] for _, pose in valid_pose_samples])
        translation_span_m = float(
            np.linalg.norm(np.max(positions, axis=0) - np.min(positions, axis=0))
        )
        rotations = np.stack([pose[:3, :3] for _, pose in valid_pose_samples])
        # trace(R_i.T @ R_j) is the flattened matrix dot product.  Computing
        # every pair in one small matrix multiply preserves the true maximum
        # angular span without a Python O(N^2) loop at live control rate.
        flattened_rotations = rotations.reshape(len(rotations), 9)
        relative_traces = flattened_rotations @ flattened_rotations.T
        pair_cosines = np.clip((relative_traces - 1.0) * 0.5, -1.0, 1.0)
        rotation_span_rad = float(np.arccos(np.min(pair_cosines)))

    minimum_span_sec = requested_window_sec * max(0.0, float(minimum_span_ratio))
    quality_reasons: list[str] = []
    if sample_count < 2:
        quality_reasons.append("insufficient_samples")
    if window_span_sec + 1e-9 < minimum_span_sec:
        quality_reasons.append("insufficient_time_span")
    if pose_valid_ratio + 1e-9 < float(minimum_pose_valid_ratio):
        quality_reasons.append("pose_valid_ratio_low")
    if velocity_valid_ratio + 1e-9 < float(minimum_velocity_valid_ratio):
        quality_reasons.append("velocity_valid_ratio_low")
    if translation_span_m is None or rotation_span_rad is None:
        quality_reasons.append("pose_span_unavailable")

    return {
        "schema": "teleop_runtime.vive_motion_window/v1",
        "requested_window_sec": requested_window_sec,
        "newest_history_time": newest_time,
        "history_clock_source": (
            None if not samples else samples[-1].get("history_clock_source")
        ),
        "sample_count": sample_count,
        "window_span_sec": window_span_sec,
        "pose_valid_ratio": pose_valid_ratio,
        "velocity_valid_ratio": velocity_valid_ratio,
        "linear_speed_p90_m_s": (
            None if not linear_speeds else float(np.percentile(linear_speeds, 90.0))
        ),
        "angular_speed_p90_rad_s": (
            None if not angular_speeds else float(np.percentile(angular_speeds, 90.0))
        ),
        "translation_span_m": translation_span_m,
        "rotation_span_rad": rotation_span_rad,
        "speed_source": "openvr_validated" if linear_speeds else "invalid",
        "quality_valid": not quality_reasons,
        "quality_reasons": quality_reasons,
    }


@dataclass(frozen=True)
class TrackerMotionIntentConfig:
    enter_linear_speed_m_s: float = 0.030
    enter_angular_speed_rad_s: float = 0.180
    enter_translation_span_m: float = 0.0025
    enter_rotation_span_rad: float = 0.025
    enter_dwell_sec: float = 0.30
    exit_linear_speed_m_s: float = 0.060
    exit_angular_speed_rad_s: float = 0.40
    exit_translation_m: float = 0.004
    exit_rotation_rad: float = float(np.deg2rad(2.0))
    exit_debounce_sec: float = 0.050
    quality_grace_sec: float = 0.10

    def __post_init__(self) -> None:
        nonnegative = (
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
            "quality_grace_sec",
        )
        for name in nonnegative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
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
                "quality_grace_sec",
            )
        }


class TrackerMotionIntentStateMachine:
    """Per-tracker ACTIVE/QUIETING/INACTIVE classifier with hysteresis."""

    def __init__(self, config: TrackerMotionIntentConfig | None = None) -> None:
        self.config = config or TrackerMotionIntentConfig()
        self.reset("initialized")

    def reset(self, reason: str = "reset") -> None:
        self._state = "UNKNOWN"
        self._quiet_since: float | None = None
        self._exit_since: float | None = None
        self._last_sample_time: float | None = None
        self._last_quality_valid_time: float | None = None
        self._inactive_anchor_pose: np.ndarray | None = None
        self._transition_reason = str(reason)
        self._evidence: dict[str, Any] = {}

    def update(
        self,
        *,
        motion_window: dict[str, Any] | None,
        tracker_pose: np.ndarray,
        clutch_pressed: bool = False,
    ) -> dict[str, Any]:
        metrics = dict(motion_window or {})
        sample_time_raw = metrics.get("newest_history_time")
        quality_valid = bool(metrics.get("quality_valid", False))
        try:
            sample_time = float(sample_time_raw)
        except (TypeError, ValueError):
            sample_time = float("nan")
        pose = np.asarray(tracker_pose, dtype=float)
        pose_valid = pose.shape == (4, 4) and np.all(np.isfinite(pose))

        if clutch_pressed:
            self.reset("clutch_pressed")
            return self._metadata(metrics)
        if not np.isfinite(sample_time) or not pose_valid:
            self.reset("motion_window_invalid")
            return self._metadata(metrics)
        if self._last_sample_time is not None and sample_time < self._last_sample_time:
            self.reset("sample_time_rewind")
        self._last_sample_time = sample_time
        if not quality_valid:
            if (
                self._last_quality_valid_time is not None
                and sample_time - self._last_quality_valid_time
                <= self.config.quality_grace_sec + 1e-9
            ):
                self._transition_reason = "motion_window_quality_grace"
                return self._metadata(metrics)
            self.reset("motion_window_invalid")
            return self._metadata(metrics)
        self._last_quality_valid_time = sample_time

        linear_speed = float(metrics["linear_speed_p90_m_s"])
        angular_speed = float(metrics["angular_speed_p90_rad_s"])
        translation_span = float(metrics["translation_span_m"])
        rotation_span = float(metrics["rotation_span_rad"])
        quiet_evidence = bool(
            linear_speed <= self.config.enter_linear_speed_m_s
            and angular_speed <= self.config.enter_angular_speed_rad_s
            and translation_span <= self.config.enter_translation_span_m
            and rotation_span <= self.config.enter_rotation_span_rad
        )
        self._evidence = {
            "quiet": quiet_evidence,
            "linear_speed_p90_m_s": linear_speed,
            "angular_speed_p90_rad_s": angular_speed,
            "translation_span_m": translation_span,
            "rotation_span_rad": rotation_span,
        }

        if self._state == "UNKNOWN":
            self._state = "QUIETING" if quiet_evidence else "ACTIVE"
            self._quiet_since = sample_time if quiet_evidence else None
            self._transition_reason = "quiet_evidence" if quiet_evidence else "motion_evidence"
        elif self._state == "ACTIVE":
            if quiet_evidence:
                self._state = "QUIETING"
                self._quiet_since = sample_time
                self._transition_reason = "quiet_evidence"
        elif self._state == "QUIETING":
            if not quiet_evidence:
                self._state = "ACTIVE"
                self._quiet_since = None
                self._transition_reason = "quiet_evidence_lost"
            elif (
                self._quiet_since is not None
                and sample_time - self._quiet_since + 1e-9 >= self.config.enter_dwell_sec
            ):
                self._state = "INACTIVE"
                self._inactive_anchor_pose = pose.copy()
                self._exit_since = None
                self._transition_reason = "enter_dwell_complete"
        else:
            assert self._state == "INACTIVE"
            anchor_translation = 0.0
            anchor_rotation = 0.0
            if self._inactive_anchor_pose is not None:
                anchor_translation = float(
                    np.linalg.norm(pose[:3, 3] - self._inactive_anchor_pose[:3, 3])
                )
                anchor_rotation = _rotation_angle_rad(
                    self._inactive_anchor_pose[:3, :3], pose[:3, :3]
                )
            speed_exit_evidence = bool(
                linear_speed >= self.config.exit_linear_speed_m_s
                or angular_speed >= self.config.exit_angular_speed_rad_s
            )
            anchor_exit_evidence = bool(
                not quiet_evidence
                and (
                    anchor_translation >= self.config.exit_translation_m
                    or anchor_rotation >= self.config.exit_rotation_rad
                )
            )
            exit_evidence = speed_exit_evidence or anchor_exit_evidence
            self._evidence.update(
                {
                    "speed_exit": speed_exit_evidence,
                    "anchor_exit": anchor_exit_evidence,
                    "anchor_translation_m": anchor_translation,
                    "anchor_rotation_rad": anchor_rotation,
                }
            )
            if exit_evidence:
                if self._exit_since is None:
                    self._exit_since = sample_time
                if (
                    sample_time - self._exit_since + 1e-9
                    >= self.config.exit_debounce_sec
                ):
                    self._state = "ACTIVE"
                    self._quiet_since = None
                    self._exit_since = None
                    self._inactive_anchor_pose = None
                    self._transition_reason = "exit_evidence_debounced"
            else:
                self._exit_since = None
                # Slowly accumulated SteamVR drift is not operator intent.  A
                # locally quiet window follows that drift, while deliberate
                # movement first violates a span/velocity enter threshold and
                # leaves this anchor fixed for the exit test above.
                if quiet_evidence:
                    self._inactive_anchor_pose = pose.copy()

        return self._metadata(metrics)

    def _metadata(self, metrics: dict[str, Any]) -> dict[str, Any]:
        sample_time = metrics.get("newest_history_time")
        quiet_dwell = 0.0
        exit_dwell = 0.0
        if isinstance(sample_time, (int, float)):
            if self._quiet_since is not None:
                quiet_dwell = max(0.0, float(sample_time) - self._quiet_since)
            if self._exit_since is not None:
                exit_dwell = max(0.0, float(sample_time) - self._exit_since)
        return {
            "schema": "teleop_runtime.tracker_motion_intent/v1",
            "state": self._state,
            "inactive": self._state == "INACTIVE",
            "hold_weight": 1.0 if self._state == "INACTIVE" else 0.0,
            "quiet_dwell_sec": quiet_dwell,
            "exit_dwell_sec": exit_dwell,
            "transition_reason": self._transition_reason,
            "evidence": dict(self._evidence),
            "motion_window": metrics,
        }

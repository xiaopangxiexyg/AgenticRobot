from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from .runtime import ObservationChannel
from .transforms import parse_pose_matrix, rotation_angle_rad, translation_distance, validate_pose_matrix


@dataclass
class MotionSafetyConfig:
    max_translation_jump_m: float = 0.10
    max_rotation_jump_deg: float = 45.0
    stale_timeout_sec: float = 0.50
    invalid_frame_limit: int = 1
    freeze_on_fault: bool = True
    minimum_control_hz: float = 15.0
    required_channels: tuple[str, ...] = ()


TeleopSafetyConfig = MotionSafetyConfig


@dataclass
class SafetyViolation:
    code: str
    message: str
    channel_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class SafetyMonitor:
    def __init__(self, config: MotionSafetyConfig | None = None):
        self.config = config if config is not None else MotionSafetyConfig()
        self._last_pose_by_channel: dict[str, np.ndarray] = {}
        self._consecutive_invalid_by_channel: dict[str, int] = {}
        self._pose_reset_pending_by_channel: set[str] = set()

    def _invalid_count_reached(self, channel_name: str) -> tuple[int, bool]:
        count = int(self._consecutive_invalid_by_channel.get(channel_name, 0)) + 1
        self._consecutive_invalid_by_channel[channel_name] = count
        limit = max(1, int(self.config.invalid_frame_limit))
        return count, count >= limit

    def _reset_invalid_count(self, channel_name: str) -> None:
        self._consecutive_invalid_by_channel[channel_name] = 0

    def request_pose_baseline_reset(self, channel_name: str) -> None:
        self._last_pose_by_channel.pop(channel_name, None)
        self._pose_reset_pending_by_channel.add(channel_name)

    def request_pose_baseline_reset_many(self, channel_names: Iterable[str]) -> None:
        for channel_name in channel_names:
            self.request_pose_baseline_reset(str(channel_name))

    def evaluate_channels(
        self, channels: Mapping[str, ObservationChannel], now: float
    ) -> list[SafetyViolation]:
        violations: list[SafetyViolation] = []
        for required in self.config.required_channels:
            if required not in channels:
                self.request_pose_baseline_reset(required)
                violations.append(
                    SafetyViolation(
                        code="missing_channel",
                        message=f"Required channel '{required}' is missing.",
                        channel_name=required,
                    )
                )

        for channel_name, channel in channels.items():
            if not channel.valid:
                if channel.critical:
                    self.request_pose_baseline_reset(channel_name)
                    invalid_count, reached_limit = self._invalid_count_reached(channel_name)
                    if reached_limit:
                        violations.append(
                            SafetyViolation(
                                code="invalid_channel",
                                message=(
                                    f"Channel '{channel_name}' is marked invalid for "
                                    f"{invalid_count} consecutive frame(s), reaching limit "
                                    f"{self.config.invalid_frame_limit}."
                                ),
                                channel_name=channel_name,
                                metadata={"invalid_count": int(invalid_count)},
                            )
                        )
                continue

            if not channel.critical:
                self._reset_invalid_count(channel_name)
                continue

            age = max(0.0, now - float(channel.timestamp))
            if age > self.config.stale_timeout_sec:
                self._reset_invalid_count(channel_name)
                self.request_pose_baseline_reset(channel_name)
                violations.append(
                    SafetyViolation(
                        code="stale_channel",
                        message=(
                            f"Channel '{channel_name}' is stale: "
                            f"{age:.3f}s > {self.config.stale_timeout_sec:.3f}s."
                        ),
                        channel_name=channel_name,
                        metadata={"age_sec": age},
                    )
                )
                continue

            if channel.capability != "pose6d":
                self._reset_invalid_count(channel_name)
                continue

            valid_pose, error = validate_pose_matrix(channel.value)
            if not valid_pose:
                self.request_pose_baseline_reset(channel_name)
                invalid_count, reached_limit = self._invalid_count_reached(channel_name)
                if reached_limit:
                    violations.append(
                        SafetyViolation(
                            code="invalid_pose",
                            message=(
                                f"Channel '{channel_name}' pose is invalid for "
                                f"{invalid_count} consecutive frame(s): {error}"
                            ),
                            channel_name=channel_name,
                            metadata={"invalid_count": int(invalid_count)},
                        )
                    )
                continue

            # A channel marked ``valid`` is not enough to break a run of malformed
            # poses.  Reset the counter only after the pose payload itself has
            # passed validation; otherwise every malformed frame starts again at
            # one and ``invalid_frame_limit > 1`` can never be reached.
            self._reset_invalid_count(channel_name)
            current_pose = parse_pose_matrix(channel.value)
            if channel_name in self._pose_reset_pending_by_channel:
                self._last_pose_by_channel[channel_name] = current_pose
                self._pose_reset_pending_by_channel.discard(channel_name)
                continue
            previous_pose = self._last_pose_by_channel.get(channel_name)
            if previous_pose is not None:
                translation_jump = translation_distance(previous_pose, current_pose)
                rotation_jump_deg = np.rad2deg(rotation_angle_rad(previous_pose, current_pose))
                if translation_jump > self.config.max_translation_jump_m:
                    violations.append(
                        SafetyViolation(
                            code="translation_jump",
                            message=(
                                f"Channel '{channel_name}' translation jump {translation_jump:.4f}m "
                                f"exceeds {self.config.max_translation_jump_m:.4f}m."
                            ),
                            channel_name=channel_name,
                            metadata={"translation_jump_m": float(translation_jump)},
                        )
                    )
                if rotation_jump_deg > self.config.max_rotation_jump_deg:
                    violations.append(
                        SafetyViolation(
                            code="rotation_jump",
                            message=(
                                f"Channel '{channel_name}' rotation jump {rotation_jump_deg:.2f}deg "
                                f"exceeds {self.config.max_rotation_jump_deg:.2f}deg."
                            ),
                            channel_name=channel_name,
                            metadata={"rotation_jump_deg": float(rotation_jump_deg)},
                        )
                    )
            self._last_pose_by_channel[channel_name] = current_pose
        return violations

    def evaluate_control_rate(self, control_hz: float) -> list[SafetyViolation]:
        if control_hz >= self.config.minimum_control_hz:
            return []
        return [
            SafetyViolation(
                code="low_control_rate",
                message=(
                    f"Control rate {control_hz:.2f}Hz is below minimum "
                    f"{self.config.minimum_control_hz:.2f}Hz."
                ),
                metadata={"control_hz": float(control_hz)},
            )
        ]


__all__ = [
    "MotionSafetyConfig",
    "SafetyMonitor",
    "SafetyViolation",
    "TeleopSafetyConfig",
]

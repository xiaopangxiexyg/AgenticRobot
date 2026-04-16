from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .schemas import ObservationChannel
from .transforms import as_pose_matrix, rotation_angle_rad, translation_distance, validate_pose_matrix


@dataclass
class TeleopSafetyConfig:
    max_translation_jump_m: float = 0.10
    max_rotation_jump_deg: float = 45.0
    max_target_speed_mps: float = 1.0
    max_target_angular_speed_dps: float = 360.0
    stale_timeout_sec: float = 0.50
    invalid_frame_limit: int = 1
    ik_failure_limit: int = 3
    freeze_on_fault: bool = True
    freeze_scope: str = "whole_rig"
    minimum_control_hz: float = 15.0
    target_control_hz: float = 30.0
    required_channels: tuple[str, ...] = ()


@dataclass
class SafetyViolation:
    code: str
    message: str
    channel_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class SafetyMonitor:
    def __init__(self, config: TeleopSafetyConfig | None = None):
        self.config = config if config is not None else TeleopSafetyConfig()
        self._last_pose_by_channel: dict[str, np.ndarray] = {}

    def evaluate_channels(
        self, channels: Mapping[str, ObservationChannel], now: float
    ) -> list[SafetyViolation]:
        violations: list[SafetyViolation] = []
        for required in self.config.required_channels:
            if required not in channels:
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
                    violations.append(
                        SafetyViolation(
                            code="invalid_channel",
                            message=f"Channel '{channel_name}' is marked invalid.",
                            channel_name=channel_name,
                        )
                    )
                continue

            if not channel.critical:
                continue

            age = max(0.0, now - float(channel.timestamp))
            if age > self.config.stale_timeout_sec:
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

            if channel.capability != "pose6d":
                continue

            valid_pose, error = validate_pose_matrix(channel.value)
            if not valid_pose:
                violations.append(
                    SafetyViolation(
                        code="invalid_pose",
                        message=f"Channel '{channel_name}' pose is invalid: {error}",
                        channel_name=channel_name,
                    )
                )
                continue

            current_pose = as_pose_matrix(channel.value)
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

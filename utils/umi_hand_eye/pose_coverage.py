from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from utils.hand_eye.transforms import as_transform, transform_distance


@dataclass(frozen=True)
class PoseCoverageConfig:
    min_samples: int = 10
    min_translation_span_m: float = 0.10
    min_rotation_span_deg: float = 30.0
    min_rotation_axis_separation_deg: float = 25.0
    min_axis_observation_rotation_deg: float = 8.0

    def __post_init__(self) -> None:
        if int(self.min_samples) < 3:
            raise ValueError("min_samples must be >= 3")
        for name in (
            "min_translation_span_m",
            "min_rotation_span_deg",
            "min_rotation_axis_separation_deg",
            "min_axis_observation_rotation_deg",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PoseCoverage:
    sample_count: int
    translation_span_m: float
    rotation_span_deg: float
    rotation_axis_separation_deg: float
    has_rotation_axis_diversity: bool
    ready: bool
    hint: str

    def overlay_lines(self, config: PoseCoverageConfig) -> list[str]:
        axis_text = (
            f"{self.rotation_axis_separation_deg:.0f}deg"
            if self.has_rotation_axis_diversity
            else "need 2 axes"
        )
        return [
            f"coverage samples {self.sample_count}/{int(config.min_samples)}",
            (
                f"span {self.translation_span_m * 1000.0:.0f}/"
                f"{float(config.min_translation_span_m) * 1000.0:.0f}mm, "
                f"{self.rotation_span_deg:.0f}/{float(config.min_rotation_span_deg):.0f}deg"
            ),
            f"rotation axes: {axis_text}",
            ("READY - press c to solve" if self.ready else f"NEXT: {self.hint}"),
        ]


def evaluate_pose_coverage(
    poses: Sequence[object],
    config: PoseCoverageConfig,
) -> PoseCoverage:
    transforms = [as_transform(value, name="tracker_pose") for value in poses]
    translation_span_m = 0.0
    rotation_span_deg = 0.0
    axes: list[np.ndarray] = []
    for first_index, first in enumerate(transforms):
        for second in transforms[first_index + 1 :]:
            translation_m, rotation_deg = transform_distance(first, second)
            translation_span_m = max(translation_span_m, float(translation_m))
            rotation_span_deg = max(rotation_span_deg, float(rotation_deg))
            relative_rotation = first[:3, :3].T @ second[:3, :3]
            rotvec = Rotation.from_matrix(relative_rotation).as_rotvec()
            angle_deg = math.degrees(float(np.linalg.norm(rotvec)))
            if angle_deg >= float(config.min_axis_observation_rotation_deg):
                axes.append(rotvec / np.linalg.norm(rotvec))

    axis_separation_deg = _max_unoriented_axis_separation_deg(axes)
    has_axis_diversity = (
        axis_separation_deg >= float(config.min_rotation_axis_separation_deg)
    )
    enough_samples = len(transforms) >= int(config.min_samples)
    enough_translation = (
        translation_span_m >= float(config.min_translation_span_m)
    )
    enough_rotation = rotation_span_deg >= float(config.min_rotation_span_deg)
    ready = bool(
        enough_samples and enough_translation and enough_rotation and has_axis_diversity
    )
    if not enough_samples:
        hint = f"collect {int(config.min_samples) - len(transforms)} more diverse poses"
    elif not enough_translation:
        hint = "move farther across the desk/workspace"
    elif not enough_rotation:
        hint = "add a larger wrist/camera rotation"
    elif not has_axis_diversity:
        hint = "rotate around a different, non-parallel axis"
    else:
        hint = "coverage complete"
    return PoseCoverage(
        sample_count=len(transforms),
        translation_span_m=float(translation_span_m),
        rotation_span_deg=float(rotation_span_deg),
        rotation_axis_separation_deg=float(axis_separation_deg),
        has_rotation_axis_diversity=bool(has_axis_diversity),
        ready=ready,
        hint=hint,
    )


def _max_unoriented_axis_separation_deg(axes: Sequence[np.ndarray]) -> float:
    maximum = 0.0
    for first_index, first in enumerate(axes):
        for second in axes[first_index + 1 :]:
            cosine = float(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0))
            maximum = max(maximum, math.degrees(math.acos(cosine)))
    return float(maximum)


__all__ = [
    "PoseCoverage",
    "PoseCoverageConfig",
    "evaluate_pose_coverage",
]

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from robot_motion.pose_smoothing import (
    SCHEMA_VERSION,
    BimanualPoseSmoother,
    PoseSmoother,
    PoseSmoothingConfig,
    PoseSmoothingResult,
    load_pose_smoothing_config,
)


DEFAULT_POSE_SMOOTHING_CONFIG_PATH = Path(__file__).resolve().parent / "pose_smoothing.json"
IMPEDANCE_POSE_SMOOTHING_CONFIG_PATH = Path(__file__).resolve().parent / "impedance_pose_smoothing.json"


def load_default_pose_smoothing_config() -> PoseSmoothingConfig:
    if DEFAULT_POSE_SMOOTHING_CONFIG_PATH.exists():
        return load_pose_smoothing_config(DEFAULT_POSE_SMOOTHING_CONFIG_PATH)
    return replace(PoseSmoothingConfig(), source_path=str(DEFAULT_POSE_SMOOTHING_CONFIG_PATH))


def load_impedance_pose_smoothing_config() -> PoseSmoothingConfig:
    """Load the force-teleop-only target smoothing profile.

    Position control deliberately keeps its established target smoothing profile.
    """

    if IMPEDANCE_POSE_SMOOTHING_CONFIG_PATH.exists():
        return load_pose_smoothing_config(IMPEDANCE_POSE_SMOOTHING_CONFIG_PATH)
    return replace(PoseSmoothingConfig(), source_path=str(IMPEDANCE_POSE_SMOOTHING_CONFIG_PATH))


def build_bimanual_pose_smoother(
    config: PoseSmoothingConfig | None = None,
) -> BimanualPoseSmoother:
    return BimanualPoseSmoother(config or load_default_pose_smoothing_config())


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_POSE_SMOOTHING_CONFIG_PATH",
    "IMPEDANCE_POSE_SMOOTHING_CONFIG_PATH",
    "BimanualPoseSmoother",
    "PoseSmoother",
    "PoseSmoothingConfig",
    "PoseSmoothingResult",
    "build_bimanual_pose_smoother",
    "load_default_pose_smoothing_config",
    "load_impedance_pose_smoothing_config",
    "load_pose_smoothing_config",
]

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from .transforms import as_pose_matrix


DEFAULT_VIVE_TRACKER_HEIGHT_CORRECTION_M = 0.04


@dataclass(frozen=True)
class ViveTrackerAlignmentConfig:
    """Centralized VIVE tracker-to-control-frame alignment defaults."""

    height_correction_enabled: bool = True
    height_correction_m: float = DEFAULT_VIVE_TRACKER_HEIGHT_CORRECTION_M
    height_correction_local_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def local_translation_offset(self, height_m: float | None = None) -> np.ndarray:
        height = self.height_correction_m if height_m is None else float(height_m)
        return np.asarray(self.height_correction_local_axis, dtype=float) * height


DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG = ViveTrackerAlignmentConfig()


# Local rotation that maps the calibrated tracker-hand control frame to the
# RM65 tool TCP action frame. The same orientation is kept for the compatibility
# arm_endpoint/hand_base frames; only their origins differ.
#
# Observed mapping:
# - tracker +X -> tcp -Z
# - tracker +Y -> tcp -Y
# - tracker +Z -> tcp -X
TRACKER_HAND_TO_RM65_ARM_ENDPOINT_ROTATION = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)

TRACKER_HAND_TO_RM65_TCP_ROTATION = TRACKER_HAND_TO_RM65_ARM_ENDPOINT_ROTATION


def tracker_hand_to_rm65_arm_endpoint_rotation() -> np.ndarray:
    """Compatibility alias for older endpoint-named call sites."""

    return TRACKER_HAND_TO_RM65_ARM_ENDPOINT_ROTATION.copy()


def tracker_hand_to_rm65_tcp_rotation() -> np.ndarray:
    return TRACKER_HAND_TO_RM65_TCP_ROTATION.copy()


def vive_tracker_height_correction_offset(
    height_m: float = DEFAULT_VIVE_TRACKER_HEIGHT_CORRECTION_M,
    *,
    config: ViveTrackerAlignmentConfig = DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
) -> np.ndarray:
    return config.local_translation_offset(height_m)


def vive_tracker_local_translation_offset_from_args(
    args: argparse.Namespace,
    *,
    config: ViveTrackerAlignmentConfig = DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
) -> np.ndarray | None:
    if not bool(
        getattr(args, "vive_tracker_height_correction", config.height_correction_enabled)
    ):
        return None
    height_m = float(
        getattr(args, "vive_tracker_height_correction_m", config.height_correction_m)
    )
    if height_m < 0.0:
        raise ValueError("--vive-tracker-height-correction-m must be >= 0.")
    return vive_tracker_height_correction_offset(height_m, config=config)


def add_vive_tracker_height_correction_args(
    parser: argparse.ArgumentParser,
    *,
    config: ViveTrackerAlignmentConfig = DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
) -> None:
    parser.add_argument(
        "--vive-tracker-height-correction",
        dest="vive_tracker_height_correction",
        action="store_true",
        help="Shift wrist tracker input by the configured local offset before arm IK. Default.",
    )
    parser.add_argument(
        "--no-vive-tracker-height-correction",
        dest="vive_tracker_height_correction",
        action="store_false",
        help=(
            "Disable the "
            f"{config.height_correction_m * 100.0:.0f}cm VIVE tracker local offset correction."
        ),
    )
    parser.add_argument(
        "--vive-tracker-height-correction-m",
        type=float,
        default=config.height_correction_m,
        help=(
            "VIVE tracker local offset distance in meters; applied along local "
            f"{list(config.height_correction_local_axis)} * value."
        ),
    )
    parser.set_defaults(vive_tracker_height_correction=config.height_correction_enabled)


def apply_local_pose_rotation(pose: Any, local_rotation: Any) -> np.ndarray:
    pose_matrix = as_pose_matrix(pose)
    rotation = np.asarray(local_rotation, dtype=float).reshape(3, 3)
    corrected = pose_matrix.copy()
    corrected[:3, :3] = corrected[:3, :3] @ rotation
    return corrected

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import as_pose_matrix


VIVE_TRACKER_ALIGNMENT_SCHEMA_VERSION = "teleop_runtime.vive_tracker_alignment.v1"
DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "vive_tracker_alignment.json"
)
_REQUIRED_GRIPPER_INPUT_SOURCES = frozenset({"q6", "angle_trigger"})


@dataclass(frozen=True)
class ViveTrackerAlignmentConfig:
    """Centralized VIVE tracker-to-control-frame alignment defaults."""

    height_correction_enabled: bool
    height_correction_local_axis: tuple[float, float, float]
    default_gripper_input_source: str
    height_correction_m_by_gripper_input_source: Mapping[str, float]
    source_path: str | None

    @property
    def height_correction_m(self) -> float:
        """Compatibility view of the configured default source correction."""

        return self.height_correction_for_gripper_input_source(
            self.default_gripper_input_source
        )

    def normalized_gripper_input_source(self, input_source: str | None) -> str:
        source = str(input_source or "").strip().lower()
        if source in {"", "auto", "none"}:
            return self.default_gripper_input_source
        if source not in self.height_correction_m_by_gripper_input_source:
            raise ValueError(
                "Unsupported gripper input source for VIVE tracker alignment: "
                f"{source!r}."
            )
        return source

    def height_correction_for_gripper_input_source(
        self,
        input_source: str | None,
    ) -> float:
        source = self.normalized_gripper_input_source(input_source)
        return float(self.height_correction_m_by_gripper_input_source[source])

    def local_translation_offset(
        self,
        height_m: float | None = None,
        *,
        gripper_input_source: str | None = None,
    ) -> np.ndarray:
        height = (
            self.height_correction_for_gripper_input_source(gripper_input_source)
            if height_m is None
            else float(height_m)
        )
        return np.asarray(self.height_correction_local_axis, dtype=float) * height


def _finite_nonnegative_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite non-negative number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite non-negative number.") from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number.")
    return number


def load_vive_tracker_alignment_config(
    path: str | Path = DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG_PATH,
) -> ViveTrackerAlignmentConfig:
    source_path = Path(path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("VIVE tracker alignment config must be a JSON object.")
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != VIVE_TRACKER_ALIGNMENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported VIVE tracker alignment schema_version: {schema_version!r}."
        )
    enabled = payload.get("height_correction_enabled")
    if not isinstance(enabled, bool):
        raise ValueError("height_correction_enabled must be a boolean.")
    axis = np.asarray(payload.get("height_correction_local_axis"), dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError(
            "height_correction_local_axis must contain three finite numbers."
        )
    if not np.isclose(float(np.linalg.norm(axis)), 1.0, atol=1e-9):
        raise ValueError("height_correction_local_axis must be a unit vector.")
    raw_by_source = payload.get("height_correction_m_by_gripper_input_source")
    if not isinstance(raw_by_source, Mapping):
        raise ValueError(
            "height_correction_m_by_gripper_input_source must be a JSON object."
        )
    missing_sources = _REQUIRED_GRIPPER_INPUT_SOURCES - {
        str(key).strip().lower() for key in raw_by_source
    }
    if missing_sources:
        raise ValueError(
            "VIVE tracker alignment config is missing gripper input source(s): "
            f"{', '.join(sorted(missing_sources))}."
        )
    by_source = {
        str(key).strip().lower(): _finite_nonnegative_float(
            value,
            field_name=(
                "height_correction_m_by_gripper_input_source."
                f"{str(key).strip().lower()}"
            ),
        )
        for key, value in raw_by_source.items()
    }
    default_source = (
        str(payload.get("default_gripper_input_source", "")).strip().lower()
    )
    if default_source not in by_source:
        raise ValueError(
            "default_gripper_input_source must name a configured input source."
        )
    return ViveTrackerAlignmentConfig(
        height_correction_enabled=enabled,
        height_correction_local_axis=tuple(float(value) for value in axis),
        default_gripper_input_source=default_source,
        height_correction_m_by_gripper_input_source=by_source,
        source_path=str(source_path),
    )


DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG = load_vive_tracker_alignment_config()
DEFAULT_VIVE_TRACKER_HEIGHT_CORRECTION_M = (
    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_m
)


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


def vive_tracker_height_correction_m_from_args(
    args: argparse.Namespace,
    *,
    config: ViveTrackerAlignmentConfig = DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
) -> float:
    configured_height_m = getattr(args, "vive_tracker_height_correction_m", None)
    if configured_height_m is None:
        configured_height_m = config.height_correction_for_gripper_input_source(
            getattr(args, "gripper_input_source", None)
        )
    return _finite_nonnegative_float(
        configured_height_m,
        field_name="--vive-tracker-height-correction-m",
    )


def vive_tracker_alignment_input_source_from_args(
    args: argparse.Namespace,
    *,
    config: ViveTrackerAlignmentConfig = DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
) -> str:
    return config.normalized_gripper_input_source(
        getattr(args, "gripper_input_source", None)
    )


def vive_tracker_local_translation_offset_from_args(
    args: argparse.Namespace,
    *,
    config: ViveTrackerAlignmentConfig = DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
) -> np.ndarray | None:
    if not bool(
        getattr(
            args, "vive_tracker_height_correction", config.height_correction_enabled
        )
    ):
        return None
    height_m = vive_tracker_height_correction_m_from_args(args, config=config)
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
        help="Disable the configured VIVE tracker local offset correction.",
    )
    parser.add_argument(
        "--vive-tracker-height-correction-m",
        type=float,
        default=None,
        help=(
            "Override the VIVE tracker local offset distance in meters. By default, "
            "the value is selected from the alignment config after resolving the "
            "gripper input source; the offset is applied along local "
            f"{list(config.height_correction_local_axis)}."
        ),
    )
    parser.set_defaults(vive_tracker_height_correction=config.height_correction_enabled)


def apply_local_pose_rotation(pose: Any, local_rotation: Any) -> np.ndarray:
    pose_matrix = as_pose_matrix(pose)
    rotation = np.asarray(local_rotation, dtype=float).reshape(3, 3)
    corrected = pose_matrix.copy()
    corrected[:3, :3] = corrected[:3, :3] @ rotation
    return corrected

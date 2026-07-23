"""Hardware-facing compatibility wrapper for the pure calibration contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.tactile_calibration import (
    BASE_FRAME_SHAPE,
    CALIBRATION_FORMAT,
    CALIBRATION_POLICIES,
    TactileCalibration,
    build_stable_base_frame,
    load_tactile_calibration as _load_tactile_calibration,
    normalize_calibration_policy,
    save_tactile_calibration as _save_tactile_calibration,
)


def default_calibration_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "tactile_calibration"


def save_tactile_calibration(
    directory: str | Path | None,
    sensor_key: Any,
    frame: Any,
    *,
    source: Mapping[str, Any] | None = None,
) -> TactileCalibration:
    return _save_tactile_calibration(
        directory or default_calibration_dir(),
        sensor_key,
        frame,
        source=source,
    )


def load_tactile_calibration(
    directory: str | Path | None,
    sensor_key: Any,
    *,
    required: bool = False,
) -> TactileCalibration | None:
    return _load_tactile_calibration(
        directory or default_calibration_dir(),
        sensor_key,
        required=required,
    )


__all__ = [
    "BASE_FRAME_SHAPE",
    "CALIBRATION_FORMAT",
    "CALIBRATION_POLICIES",
    "TactileCalibration",
    "build_stable_base_frame",
    "default_calibration_dir",
    "load_tactile_calibration",
    "normalize_calibration_policy",
    "save_tactile_calibration",
]

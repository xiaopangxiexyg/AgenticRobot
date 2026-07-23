"""Pure DM-Tac calibration records and file validation.

This module deliberately has no knowledge of a hardware package or station
default directory.  Hardware and offline callers select their own calibration
provider/directory and share only the immutable file contract implemented here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CALIBRATION_FORMAT = "dexteleop_dmtac_base_frame_v1"
CALIBRATION_POLICIES = ("auto", "required", "off")
BASE_FRAME_SHAPE = (480, 640)


def normalize_calibration_policy(value: str | None) -> str:
    policy = str(value or "auto").strip().lower()
    if policy not in CALIBRATION_POLICIES:
        raise ValueError(
            f"calibration policy must be one of {CALIBRATION_POLICIES}, got {value!r}"
        )
    return policy


def _sensor_token(sensor_key: Any) -> str:
    text = str(sensor_key).strip()
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    if not text or not token:
        raise ValueError("sensor_key must be non-empty")
    return token


def _validated_frame(value: Any) -> np.ndarray:
    frame = np.asarray(value)
    if frame.shape != BASE_FRAME_SHAPE:
        raise ValueError(
            f"DM-Tac base frame must have shape {BASE_FRAME_SHAPE}, got {frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise ValueError(f"DM-Tac base frame must be uint8, got {frame.dtype}")
    return np.ascontiguousarray(frame)


def _sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(frame))).hexdigest()


def build_stable_base_frame(
    frames: Sequence[np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a lossless uint8 median base and report temporal deviation."""

    if not frames:
        raise ValueError("at least one raw frame is required")
    stack = np.stack([_validated_frame(frame) for frame in frames])
    base = np.rint(np.median(stack, axis=0)).astype(np.uint8)
    deviation = np.abs(stack.astype(np.int16) - base.astype(np.int16))
    metrics = {
        "mean_abs_deviation": float(np.mean(deviation)),
        "p99_abs_deviation": float(np.percentile(deviation, 99.0)),
        "max_abs_deviation": float(np.max(deviation)),
    }
    return np.ascontiguousarray(base), metrics


@dataclass(frozen=True)
class TactileCalibration:
    sensor_key: str
    frame: np.ndarray
    metadata: dict[str, Any]
    metadata_path: Path
    frame_path: Path

    @property
    def sha256(self) -> str:
        return str(self.metadata["sha256"])

    def descriptor(self) -> dict[str, Any]:
        return {
            "format": CALIBRATION_FORMAT,
            "sensor_key": self.sensor_key,
            "sha256": self.sha256,
            "created_at": self.metadata.get("created_at"),
            "source": self.metadata.get("source"),
            "metadata_path": str(self.metadata_path),
        }


def save_tactile_calibration(
    directory: str | Path,
    sensor_key: Any,
    frame: Any,
    *,
    source: Mapping[str, Any] | None = None,
) -> TactileCalibration:
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sensor = str(sensor_key).strip()
    token = _sensor_token(sensor)
    array = _validated_frame(frame)
    frame_path = root / f"{token}.base_frame.npy"
    metadata_path = root / f"{token}.json"
    frame_tmp = frame_path.with_suffix(frame_path.suffix + ".tmp")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with frame_tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    frame_tmp.replace(frame_path)
    metadata = {
        "format": CALIBRATION_FORMAT,
        "sensor_key": sensor,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frame_file": frame_path.name,
        "shape": list(BASE_FRAME_SHAPE),
        "dtype": "uint8",
        "sha256": _sha256(array),
        "source": dict(source or {}),
    }
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata_tmp.replace(metadata_path)
    return TactileCalibration(sensor, array, metadata, metadata_path, frame_path)


def load_tactile_calibration(
    directory: str | Path,
    sensor_key: Any,
    *,
    required: bool = False,
) -> TactileCalibration | None:
    root = Path(directory).expanduser().resolve()
    sensor = str(sensor_key).strip()
    metadata_path = root / f"{_sensor_token(sensor)}.json"
    if not metadata_path.is_file():
        if required:
            raise FileNotFoundError(
                f"No fixed DM-Tac calibration for sensor {sensor!r} in {root}"
            )
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("format") != CALIBRATION_FORMAT:
        raise ValueError(f"Unsupported DM-Tac calibration metadata: {metadata_path}")
    if str(metadata.get("sensor_key")) != sensor:
        raise ValueError(
            f"Calibration sensor mismatch in {metadata_path}: expected {sensor!r}"
        )
    frame_name = Path(str(metadata.get("frame_file") or ""))
    if frame_name.name != str(frame_name) or not frame_name.name:
        raise ValueError(f"Invalid calibration frame_file in {metadata_path}")
    frame_path = metadata_path.parent / frame_name
    with frame_path.open("rb") as handle:
        frame = _validated_frame(np.load(handle, allow_pickle=False))
    digest = _sha256(frame)
    if digest != str(metadata.get("sha256") or ""):
        raise ValueError(f"DM-Tac calibration checksum mismatch: {frame_path}")
    return TactileCalibration(sensor, frame, metadata, metadata_path, frame_path)


__all__ = [
    "BASE_FRAME_SHAPE",
    "CALIBRATION_FORMAT",
    "CALIBRATION_POLICIES",
    "TactileCalibration",
    "build_stable_base_frame",
    "load_tactile_calibration",
    "normalize_calibration_policy",
    "save_tactile_calibration",
]

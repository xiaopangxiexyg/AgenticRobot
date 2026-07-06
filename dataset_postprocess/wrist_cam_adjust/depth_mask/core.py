from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


WRIST_DEPTH_MASK_FORMAT = "dexteleop_wrist_depth_mask_bitpack_v1"
WRIST_DEPTH_RAW_FORMAT = "dexteleop_wrist_depth_raw_z16_v1"


@dataclass(frozen=True)
class DepthRangeMaskConfig:
    depth_min_m: float = 0.05
    depth_max_m: float = 0.5

    def __post_init__(self) -> None:
        if float(self.depth_min_m) < 0.0:
            raise ValueError("depth_min_m must be >= 0.")
        if float(self.depth_max_m) <= float(self.depth_min_m):
            raise ValueError("depth_max_m must be greater than depth_min_m.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_min_m": float(self.depth_min_m),
            "depth_max_m": float(self.depth_max_m),
        }


@dataclass(frozen=True)
class DepthRangeMaskResult:
    mask: np.ndarray
    valid_fraction: float
    foreground_fraction: float
    depth_min_m: float
    depth_max_m: float
    depth_scale_m_per_unit: float

    def stats_dict(self) -> dict[str, Any]:
        return {
            "valid_fraction": float(self.valid_fraction),
            "foreground_fraction": float(self.foreground_fraction),
            "depth_min_m": float(self.depth_min_m),
            "depth_max_m": float(self.depth_max_m),
            "depth_scale_m_per_unit": float(self.depth_scale_m_per_unit),
        }


def depth_range_mask(
    depth_z16: Any,
    *,
    depth_scale_m_per_unit: float,
    config: DepthRangeMaskConfig | None = None,
) -> DepthRangeMaskResult:
    """Return a 0/1 mask for depth pixels inside the configured metric range.

    Collection uses this as cheap depth evidence only. Geometry-aware object/table
    reasoning, temporal hold, and hole repair should happen in wrist_cam_adjust
    post-processing where camera poses and task-specific parameters are available.
    """

    cfg = config if config is not None else DepthRangeMaskConfig()
    depth = np.asarray(depth_z16)
    if depth.ndim != 2:
        raise ValueError(f"depth_z16 must be a 2D array, got shape={depth.shape!r}.")
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)
    scale = float(depth_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"depth_scale_m_per_unit must be positive, got {depth_scale_m_per_unit!r}.")

    depth_m = depth.astype(np.float32, copy=False) * np.float32(scale)
    valid = depth > 0
    in_range = (
        valid
        & (depth_m >= float(cfg.depth_min_m))
        & (depth_m <= float(cfg.depth_max_m))
    )
    total = int(depth.size)
    valid_count = int(np.count_nonzero(valid))
    foreground_count = int(np.count_nonzero(in_range))
    return DepthRangeMaskResult(
        mask=in_range.astype(np.uint8, copy=False),
        valid_fraction=0.0 if total <= 0 else float(valid_count) / float(total),
        foreground_fraction=0.0 if total <= 0 else float(foreground_count) / float(total),
        depth_min_m=float(cfg.depth_min_m),
        depth_max_m=float(cfg.depth_max_m),
        depth_scale_m_per_unit=scale,
    )


def pack_binary_mask(mask: Any) -> bytes:
    """Pack a 2D boolean/0-1 mask to bytes using numpy's stable bit order."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be a 2D array, got shape={array.shape!r}.")
    binary = (array > 0).astype(np.uint8, copy=False).reshape(-1)
    return np.packbits(binary).tobytes(order="C")


def unpack_binary_mask(data: bytes | bytearray | memoryview, *, height: int, width: int) -> np.ndarray:
    """Unpack one bit-packed mask frame into a uint8 0/1 image."""

    h = int(height)
    w = int(width)
    if h <= 0 or w <= 0:
        raise ValueError("height and width must be positive.")
    bit_count = h * w
    expected_bytes = (bit_count + 7) // 8
    payload = bytes(data)
    if len(payload) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} packed bytes, got {len(payload)}.")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), count=bit_count)
    return bits.reshape(h, w).astype(np.uint8, copy=False)


def mask_manifest_payload(
    *,
    camera_key: str,
    relative_dir: str,
    stream_path: str,
    frame_count: int,
    height: int,
    width: int,
    records: list[Mapping[str, Any]],
    config: DepthRangeMaskConfig,
) -> dict[str, Any]:
    return {
        "format": WRIST_DEPTH_MASK_FORMAT,
        "camera_key": str(camera_key),
        "relative_dir": str(relative_dir),
        "stream": {
            "path": str(stream_path),
            "encoding": "np.packbits",
            "dtype": "uint1",
            "shape": [int(height), int(width)],
            "packed_bytes_per_frame": int((int(height) * int(width) + 7) // 8),
            "frame_count": int(frame_count),
        },
        "semantics": {
            "name": "depth_range_mask",
            "value_1": "aligned_depth_z16 is within [depth_min_m, depth_max_m]",
            "value_0": "depth invalid or outside configured range",
            "postprocess_note": (
                "Hole filling, temporal anti-flicker, and table/object reasoning are "
                "performed by dataset_postprocess.wrist_cam_adjust post-processing."
            ),
        },
        "config": config.to_dict(),
        "frames": [dict(record) for record in records],
    }


def raw_depth_manifest_payload(
    *,
    camera_key: str,
    relative_dir: str,
    stream_path: str,
    frame_count: int,
    height: int,
    width: int,
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "format": WRIST_DEPTH_RAW_FORMAT,
        "camera_key": str(camera_key),
        "relative_dir": str(relative_dir),
        "stream": {
            "path": str(stream_path),
            "encoding": "raw_c_order",
            "dtype": "uint16",
            "shape": [int(height), int(width)],
            "bytes_per_frame": int(int(height) * int(width) * np.dtype(np.uint16).itemsize),
            "frame_count": int(frame_count),
        },
        "semantics": {
            "name": "aligned_depth_z16",
            "unit": "RealSense z16 units; multiply by frame depth_scale_m_per_unit",
            "value_0": "invalid depth",
            "debug_only": True,
        },
        "frames": [dict(record) for record in records],
    }

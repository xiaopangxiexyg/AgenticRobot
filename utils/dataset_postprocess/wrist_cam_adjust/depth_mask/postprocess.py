from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from utils.dataset_postprocess.real_undistort.intrinsics import require_cv2

from .core import (
    DEPTH_MASK_FAMILY,
    DEPTH_RANGE_POSTPROCESSED_MASK_NAME,
    WRIST_DEPTH_MASK_FORMAT,
    pack_binary_mask,
)
from .visualize import MaskStream, iter_mask_frames


@dataclass(frozen=True)
class DepthMaskPostprocessConfig:
    close_radius_px: int = 5
    open_radius_px: int = 0
    median_ksize: int = 5
    fill_hole_max_area_px: int = 4096
    min_component_area_px: int = 0
    gaussian_blur_px: int = 0
    gaussian_threshold: float = 0.5
    temporal_gap_fill_frames: int = 2
    temporal_majority_window: int = 0
    temporal_majority_threshold: float = 0.5

    def __post_init__(self) -> None:
        if int(self.close_radius_px) < 0:
            raise ValueError("close_radius_px must be >= 0.")
        if int(self.open_radius_px) < 0:
            raise ValueError("open_radius_px must be >= 0.")
        if int(self.median_ksize) < 0:
            raise ValueError("median_ksize must be >= 0.")
        if int(self.fill_hole_max_area_px) < 0:
            raise ValueError("fill_hole_max_area_px must be >= 0.")
        if int(self.min_component_area_px) < 0:
            raise ValueError("min_component_area_px must be >= 0.")
        if int(self.gaussian_blur_px) < 0:
            raise ValueError("gaussian_blur_px must be >= 0.")
        if not (0.0 < float(self.gaussian_threshold) <= 1.0):
            raise ValueError("gaussian_threshold must be in (0, 1].")
        if int(self.temporal_gap_fill_frames) < 0:
            raise ValueError("temporal_gap_fill_frames must be >= 0.")
        if int(self.temporal_majority_window) < 0:
            raise ValueError("temporal_majority_window must be >= 0.")
        if not (0.0 < float(self.temporal_majority_threshold) <= 1.0):
            raise ValueError("temporal_majority_threshold must be in (0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return {
            "close_radius_px": int(self.close_radius_px),
            "open_radius_px": int(self.open_radius_px),
            "median_ksize": int(self.median_ksize),
            "fill_hole_max_area_px": int(self.fill_hole_max_area_px),
            "min_component_area_px": int(self.min_component_area_px),
            "gaussian_blur_px": int(self.gaussian_blur_px),
            "gaussian_threshold": float(self.gaussian_threshold),
            "temporal_gap_fill_frames": int(self.temporal_gap_fill_frames),
            "temporal_majority_window": int(self.temporal_majority_window),
            "temporal_majority_threshold": float(self.temporal_majority_threshold),
        }


@dataclass(frozen=True)
class DepthMaskPostprocessResult:
    manifest_path: Path
    stream_path: Path
    frame_count: int
    height: int
    width: int
    input_foreground_fraction_mean: float
    spatial_foreground_fraction_mean: float
    output_foreground_fraction_mean: float
    changed_fraction_mean: float
    temporal_changed_fraction_mean: float
    input_frame_change_fraction_mean: float
    spatial_frame_change_fraction_mean: float
    output_frame_change_fraction_mean: float
    source_declared_bytes: int
    source_actual_bytes: int
    source_trailing_bytes: int
    rollback_dir: Path | None
    config: DepthMaskPostprocessConfig = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "stream_path": str(self.stream_path),
            "frame_count": int(self.frame_count),
            "height": int(self.height),
            "width": int(self.width),
            "input_foreground_fraction_mean": float(self.input_foreground_fraction_mean),
            "spatial_foreground_fraction_mean": float(self.spatial_foreground_fraction_mean),
            "output_foreground_fraction_mean": float(self.output_foreground_fraction_mean),
            "changed_fraction_mean": float(self.changed_fraction_mean),
            "temporal_changed_fraction_mean": float(self.temporal_changed_fraction_mean),
            "input_frame_change_fraction_mean": float(self.input_frame_change_fraction_mean),
            "spatial_frame_change_fraction_mean": float(self.spatial_frame_change_fraction_mean),
            "output_frame_change_fraction_mean": float(self.output_frame_change_fraction_mean),
            "source_declared_bytes": int(self.source_declared_bytes),
            "source_actual_bytes": int(self.source_actual_bytes),
            "source_trailing_bytes": int(self.source_trailing_bytes),
            "rollback_dir": None if self.rollback_dir is None else str(self.rollback_dir),
            "config": self.config.to_dict(),
        }


def _binary_u8(mask: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)


def _odd_kernel_size(value: int) -> int:
    size = int(value)
    if size <= 1:
        return 0
    return size if size % 2 == 1 else size + 1


def _ellipse_kernel(radius_px: int) -> np.ndarray | None:
    radius = int(radius_px)
    if radius <= 0:
        return None
    cv2 = require_cv2()
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _fill_small_holes(mask_u8: np.ndarray, *, max_area_px: int) -> np.ndarray:
    max_area = int(max_area_px)
    if max_area <= 0:
        return mask_u8
    cv2 = require_cv2()
    out = _binary_u8(mask_u8)
    contours, hierarchy = cv2.findContours(out, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return out
    hierarchy_array = hierarchy.reshape(-1, 4)
    for index, contour in enumerate(contours):
        parent = int(hierarchy_array[index][3])
        if parent < 0:
            continue
        area = float(cv2.contourArea(contour))
        if area <= float(max_area):
            cv2.drawContours(out, contours, index, 255, thickness=-1)
    return out


def _remove_small_components(mask_u8: np.ndarray, *, min_area_px: int) -> np.ndarray:
    min_area = int(min_area_px)
    if min_area <= 0:
        return mask_u8
    cv2 = require_cv2()
    binary = _binary_u8(mask_u8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for label in range(1, int(count)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == label] = 255
    return out


def postprocess_depth_mask(mask: np.ndarray, config: DepthMaskPostprocessConfig | None = None) -> np.ndarray:
    """Fill holes and smooth a saved 0/1 wrist depth mask frame.

    This operates on the saved bit-packed mask, not raw metric depth. The
    current 2026-07-05 dataset did not save raw depth, so this is the first
    useful post-processing layer for the available data.
    """

    cfg = config or DepthMaskPostprocessConfig()
    cv2 = require_cv2()
    out = _binary_u8(mask)

    close_kernel = _ellipse_kernel(cfg.close_radius_px)
    if close_kernel is not None:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, close_kernel)

    out = _fill_small_holes(out, max_area_px=cfg.fill_hole_max_area_px)

    median_ksize = _odd_kernel_size(cfg.median_ksize)
    if median_ksize > 1:
        out = cv2.medianBlur(out, median_ksize)
        out = _binary_u8(out)

    open_kernel = _ellipse_kernel(cfg.open_radius_px)
    if open_kernel is not None:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, open_kernel)

    out = _remove_small_components(out, min_area_px=cfg.min_component_area_px)

    gaussian_ksize = _odd_kernel_size(cfg.gaussian_blur_px)
    if gaussian_ksize > 1:
        blurred = cv2.GaussianBlur(out, (gaussian_ksize, gaussian_ksize), 0.0)
        threshold = int(round(float(cfg.gaussian_threshold) * 255.0))
        out = np.where(blurred >= threshold, 255, 0).astype(np.uint8)

    return (out > 0).astype(np.uint8, copy=False)


def _fill_temporal_gaps(frames: np.ndarray, *, max_gap_frames: int) -> np.ndarray:
    max_gap = int(max_gap_frames)
    base = np.asarray(frames, dtype=bool)
    if max_gap <= 0 or base.shape[0] <= 2:
        return base
    out = base.copy()
    frame_count = int(base.shape[0])
    for gap_len in range(1, max_gap + 1):
        span = gap_len + 1
        if frame_count <= span:
            break
        endpoints = base[:-span] & base[span:]
        for offset in range(1, span):
            out[offset : offset + endpoints.shape[0]] |= endpoints
    return out


def _temporal_majority(frames: np.ndarray, *, window: int, threshold: float) -> np.ndarray:
    window_size = _odd_kernel_size(int(window))
    base = np.asarray(frames, dtype=np.uint8)
    if window_size <= 1 or base.shape[0] <= 1:
        return base.astype(bool, copy=False)
    radius = window_size // 2
    cumulative = np.concatenate(
        [
            np.zeros((1,) + base.shape[1:], dtype=np.uint16),
            np.cumsum(base, axis=0, dtype=np.uint16),
        ],
        axis=0,
    )
    out = np.zeros_like(base, dtype=bool)
    frame_count = int(base.shape[0])
    for frame_index in range(frame_count):
        start = max(0, frame_index - radius)
        end = min(frame_count, frame_index + radius + 1)
        votes = cumulative[end] - cumulative[start]
        required = max(1, int(np.ceil(float(end - start) * float(threshold))))
        out[frame_index] = votes >= required
    return out


def postprocess_depth_mask_sequence(
    masks: np.ndarray,
    config: DepthMaskPostprocessConfig | None = None,
) -> np.ndarray:
    """Apply temporal continuity filters to already spatially postprocessed masks."""

    cfg = config or DepthMaskPostprocessConfig()
    out = np.asarray(masks, dtype=bool)
    out = _fill_temporal_gaps(out, max_gap_frames=cfg.temporal_gap_fill_frames)
    out = _temporal_majority(
        out,
        window=cfg.temporal_majority_window,
        threshold=cfg.temporal_majority_threshold,
    )
    return out.astype(np.uint8, copy=False)


def _frame_change_fraction_mean(frames: np.ndarray) -> float:
    array = np.asarray(frames, dtype=bool)
    if array.shape[0] < 2:
        return 0.0
    changed = array[1:] != array[:-1]
    pixels_per_frame = int(np.prod(array.shape[1:]))
    if pixels_per_frame <= 0:
        return 0.0
    return float(np.count_nonzero(changed)) / float((array.shape[0] - 1) * pixels_per_frame)


def _manifest_payload(
    *,
    stream: MaskStream,
    output_dir: Path,
    stream_path: Path,
    frame_count: int,
    records: list[Mapping[str, Any]],
    config: DepthMaskPostprocessConfig,
    source_declared_bytes: int,
    source_actual_bytes: int,
    source_trailing_bytes: int,
    rollback_dir: Path | None,
) -> dict[str, Any]:
    source_manifest = None if stream.manifest_path is None else str(stream.manifest_path)
    source_stream = str(stream.stream_path)
    relative_stream = stream_path.name
    return {
        "format": WRIST_DEPTH_MASK_FORMAT,
        "camera_key": str(stream.label),
        "relative_dir": str(output_dir),
        "mask_family": DEPTH_MASK_FAMILY,
        "mask_name": DEPTH_RANGE_POSTPROCESSED_MASK_NAME,
        "stream": {
            "path": relative_stream,
            "encoding": "np.packbits",
            "dtype": "uint1",
            "shape": [int(stream.height), int(stream.width)],
            "packed_bytes_per_frame": int((int(stream.height) * int(stream.width) + 7) // 8),
            "frame_count": int(frame_count),
        },
        "semantics": {
            "name": DEPTH_RANGE_POSTPROCESSED_MASK_NAME,
            "value_1": "postprocessed foreground from saved wrist depth range mask",
            "value_0": "background after postprocess",
            "source_note": "Post-processing starts from the saved bit-packed depth range mask.",
        },
        "source": {
            "manifest_path": source_manifest,
            "stream_path": source_stream,
            "label": str(stream.label),
            "frame_count": int(stream.frame_count),
            "integrity": {
                "declared_bytes": int(source_declared_bytes),
                "actual_bytes": int(source_actual_bytes),
                "trailing_bytes": int(source_trailing_bytes),
                "read_policy": "exact_declared_frames_v1",
                "source_modified": False,
            },
        },
        "publication": {
            "mode": "atomic_directory_replace_v1",
            "rollback_dir": None if rollback_dir is None else str(rollback_dir),
        },
        "postprocess": config.to_dict(),
        "frames": [dict(record) for record in records],
    }


def process_mask_stream(
    stream: MaskStream,
    output_dir: str | Path,
    *,
    config: DepthMaskPostprocessConfig | None = None,
    max_frames: int = 0,
    overwrite: bool = False,
) -> DepthMaskPostprocessResult:
    cfg = config or DepthMaskPostprocessConfig()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass overwrite=True: {out_dir}")
    transaction_id = uuid.uuid4().hex
    staging_dir = out_dir.parent / f".{out_dir.name}.tmp-{transaction_id}"
    staging_dir.mkdir(parents=False, exist_ok=False)
    staging_stream_path = staging_dir / "mask_postprocessed.bitpack.bin"
    staging_manifest_path = staging_dir / "manifest.json"
    final_stream_path = out_dir / staging_stream_path.name
    final_manifest_path = out_dir / staging_manifest_path.name
    rollback_dir = (
        out_dir.parent / f".{out_dir.name}.rollback-{transaction_id}"
        if out_dir.exists()
        else None
    )

    frame_limit = int(stream.frame_count)
    if int(max_frames) > 0:
        frame_limit = min(frame_limit, int(max_frames))
    source_declared_bytes = int(stream.frame_count) * int(
        stream.packed_bytes_per_frame
    )
    source_actual_bytes = int(stream.stream_path.stat().st_size)
    if source_actual_bytes < source_declared_bytes:
        raise ValueError(
            "source mask stream is shorter than its manifest contract: "
            f"actual={source_actual_bytes} declared={source_declared_bytes} "
            f"path={stream.stream_path}"
        )
    source_trailing_bytes = source_actual_bytes - source_declared_bytes
    if source_trailing_bytes % int(stream.packed_bytes_per_frame):
        raise ValueError(
            "source mask stream has a partial trailing frame: "
            f"trailing_bytes={source_trailing_bytes} "
            f"packed_bytes_per_frame={stream.packed_bytes_per_frame} "
            f"path={stream.stream_path}"
        )
    input_fraction_sum = 0.0
    spatial_fraction_sum = 0.0
    output_fraction_sum = 0.0
    changed_fraction_sum = 0.0
    temporal_changed_fraction_sum = 0.0
    input_frames: list[np.ndarray] = []
    spatial_frames: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []

    for frame_index, mask in enumerate(iter_mask_frames(stream, max_frames=frame_limit)):
        processed = postprocess_depth_mask(mask, cfg)
        input_frames.append((np.asarray(mask) > 0).astype(np.uint8, copy=False))
        spatial_frames.append((np.asarray(processed) > 0).astype(np.uint8, copy=False))
        source_record: dict[str, Any] = {}
        if 0 <= frame_index < len(stream.frames):
            source_record = dict(stream.frames[frame_index])
        source_records.append(source_record)
    if len(input_frames) != frame_limit:
        raise ValueError(
            "source mask stream ended before the declared frame limit: "
            f"actual={len(input_frames)} expected={frame_limit} "
            f"path={stream.stream_path}"
        )

    records: list[dict[str, Any]] = []
    if input_frames:
        input_stack = np.stack(input_frames, axis=0).astype(bool, copy=False)
        spatial_stack = np.stack(spatial_frames, axis=0).astype(bool, copy=False)
        output_stack = postprocess_depth_mask_sequence(spatial_stack, cfg).astype(bool, copy=False)
    else:
        input_stack = np.zeros((0, int(stream.height), int(stream.width)), dtype=bool)
        spatial_stack = input_stack
        output_stack = input_stack

    with staging_stream_path.open("wb") as handle:
        for frame_index in range(int(output_stack.shape[0])):
            input_binary = input_stack[frame_index]
            spatial_binary = spatial_stack[frame_index]
            output_binary = output_stack[frame_index]
            total = int(input_binary.size)
            input_fraction = 0.0 if total <= 0 else float(np.count_nonzero(input_binary)) / float(total)
            spatial_fraction = 0.0 if total <= 0 else float(np.count_nonzero(spatial_binary)) / float(total)
            output_fraction = 0.0 if total <= 0 else float(np.count_nonzero(output_binary)) / float(total)
            changed_fraction = 0.0 if total <= 0 else float(np.count_nonzero(input_binary != output_binary)) / float(total)
            temporal_changed_fraction = (
                0.0
                if total <= 0
                else float(np.count_nonzero(spatial_binary != output_binary)) / float(total)
            )
            input_fraction_sum += input_fraction
            spatial_fraction_sum += spatial_fraction
            output_fraction_sum += output_fraction
            changed_fraction_sum += changed_fraction
            temporal_changed_fraction_sum += temporal_changed_fraction
            record: dict[str, Any] = {
                "frame_index": int(frame_index),
                "input_foreground_fraction": input_fraction,
                "spatial_foreground_fraction": spatial_fraction,
                "foreground_fraction": output_fraction,
                "changed_fraction": changed_fraction,
                "temporal_changed_fraction": temporal_changed_fraction,
            }
            source_record = source_records[frame_index]
            for key in (
                "sample_frame_index",
                "depth_frame_index",
                "source_timestamp",
                "receive_timestamp",
                "source_frame_index",
                "depth_camera_timestamp_sec",
                "depth_sensor_timestamp_ms",
                "depth_frame_number",
                "depth_scale_m_per_unit",
                "valid_fraction",
            ):
                if key in source_record:
                    record[key] = source_record[key]
            records.append(record)
            handle.write(pack_binary_mask(output_binary))
        handle.flush()
        os.fsync(handle.fileno())

    frame_count = len(records)
    payload = _manifest_payload(
        stream=stream,
        output_dir=out_dir,
        stream_path=final_stream_path,
        frame_count=frame_count,
        records=records,
        config=cfg,
        source_declared_bytes=source_declared_bytes,
        source_actual_bytes=source_actual_bytes,
        source_trailing_bytes=source_trailing_bytes,
        rollback_dir=rollback_dir,
    )
    expected_output_bytes = frame_count * int(stream.packed_bytes_per_frame)
    actual_output_bytes = int(staging_stream_path.stat().st_size)
    if actual_output_bytes != expected_output_bytes:
        raise RuntimeError(
            "postprocessed mask stream size differs from its frame contract: "
            f"actual={actual_output_bytes} expected={expected_output_bytes}"
        )
    with staging_manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=True))
        handle.flush()
        os.fsync(handle.fileno())
    if rollback_dir is not None:
        os.replace(out_dir, rollback_dir)
    try:
        os.replace(staging_dir, out_dir)
    except BaseException:
        if rollback_dir is not None and rollback_dir.exists() and not out_dir.exists():
            os.replace(rollback_dir, out_dir)
        raise
    return DepthMaskPostprocessResult(
        manifest_path=final_manifest_path,
        stream_path=final_stream_path,
        frame_count=frame_count,
        height=int(stream.height),
        width=int(stream.width),
        input_foreground_fraction_mean=0.0 if frame_count <= 0 else input_fraction_sum / float(frame_count),
        spatial_foreground_fraction_mean=0.0 if frame_count <= 0 else spatial_fraction_sum / float(frame_count),
        output_foreground_fraction_mean=0.0 if frame_count <= 0 else output_fraction_sum / float(frame_count),
        changed_fraction_mean=0.0 if frame_count <= 0 else changed_fraction_sum / float(frame_count),
        temporal_changed_fraction_mean=0.0 if frame_count <= 0 else temporal_changed_fraction_sum / float(frame_count),
        input_frame_change_fraction_mean=_frame_change_fraction_mean(input_stack),
        spatial_frame_change_fraction_mean=_frame_change_fraction_mean(spatial_stack),
        output_frame_change_fraction_mean=_frame_change_fraction_mean(output_stack),
        source_declared_bytes=source_declared_bytes,
        source_actual_bytes=source_actual_bytes,
        source_trailing_bytes=source_trailing_bytes,
        rollback_dir=rollback_dir,
        config=cfg,
    )

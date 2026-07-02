from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


try:
    import cv2
except ImportError:  # pragma: no cover - depends on runtime env
    cv2 = None


@dataclass
class TactileDenoiseConfig:
    enabled: bool = True
    mode: str = "delta"
    baseline_frames: int = 20
    deadband_abs: float = 5e-5
    temporal_alpha: float = 0.85
    spatial_blur_ksize: int = 5


def as_distributed_force_array(distributed_force: Any) -> np.ndarray:
    arr = np.asarray(distributed_force, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 2:
        raise ValueError("distributed_force must have shape (H, W, C) with at least two channels")
    return arr


class DistributedForceDenoiser:
    """Runtime denoiser for distributed_force returned by DM-Tac."""

    def __init__(self, config: TactileDenoiseConfig | None = None) -> None:
        self.config = config or TactileDenoiseConfig()
        self._prev_raw: Dict[Any, np.ndarray] = {}
        self._filtered: Dict[Any, np.ndarray] = {}
        self._baseline_sum: Dict[Any, np.ndarray] = {}
        self._baseline_count: Dict[Any, int] = {}
        self._baseline: Dict[Any, np.ndarray] = {}

    def reset(self) -> None:
        self._prev_raw.clear()
        self._filtered.clear()
        self._baseline_sum.clear()
        self._baseline_count.clear()
        self._baseline.clear()

    def transform(self, device_id: Any, distributed_force: Any) -> np.ndarray:
        raw = as_distributed_force_array(distributed_force)
        if not self.config.enabled:
            self._prev_raw[device_id] = raw.copy()
            return raw

        force = self._mode_force(device_id, raw)
        force = self._smooth(force, device_id)
        self._prev_raw[device_id] = raw.copy()
        return force

    def _mode_force(self, device_id: Any, raw: np.ndarray) -> np.ndarray:
        mode = str(self.config.mode).strip().lower()
        if mode == "absolute":
            return raw.astype(np.float32, copy=True)

        if mode == "frame_delta":
            prev = self._prev_raw.get(device_id)
            if prev is None or prev.shape != raw.shape:
                return np.zeros_like(raw, dtype=np.float32)
            return raw - prev

        if mode != "delta":
            raise ValueError("TactileDenoiseConfig.mode must be 'delta', 'absolute', or 'frame_delta'")

        baseline = self._baseline.get(device_id)
        if baseline is not None and baseline.shape == raw.shape:
            return raw - baseline

        need = max(1, int(self.config.baseline_frames))
        if device_id not in self._baseline_sum or self._baseline_sum[device_id].shape != raw.shape:
            self._baseline_sum[device_id] = np.zeros_like(raw, dtype=np.float64)
            self._baseline_count[device_id] = 0
        self._baseline_sum[device_id] += raw.astype(np.float64, copy=False)
        self._baseline_count[device_id] += 1
        count = self._baseline_count[device_id]
        if count >= need:
            self._baseline[device_id] = (self._baseline_sum[device_id] / float(count)).astype(np.float32)
            return raw - self._baseline[device_id]
        return np.zeros_like(raw, dtype=np.float32)

    def _smooth(self, force: np.ndarray, device_id: Any) -> np.ndarray:
        out = force.astype(np.float32, copy=True)

        blur_ksize = int(self.config.spatial_blur_ksize)
        if blur_ksize > 1 and cv2 is not None:
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            out = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)

        alpha = float(self.config.temporal_alpha)
        if 0.0 < alpha < 1.0:
            prev = self._filtered.get(device_id)
            if prev is not None and prev.shape == out.shape:
                out = prev * (1.0 - alpha) + out * alpha
        self._filtered[device_id] = out.copy()

        deadband = max(0.0, float(self.config.deadband_abs))
        if deadband > 0.0:
            xy = out[..., :2]
            mag = np.linalg.norm(xy, axis=2)
            xy[mag < deadband] = 0.0
            out[..., :2] = xy

        return out


def distributed_force_xy_stats(distributed_force: Any) -> Tuple[float, float]:
    force = as_distributed_force_array(distributed_force)
    xy = np.nan_to_num(force[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
    mag = np.linalg.norm(xy, axis=2)
    return float(np.nanmean(mag)) if mag.size else 0.0, float(np.nanmax(mag)) if mag.size else 0.0

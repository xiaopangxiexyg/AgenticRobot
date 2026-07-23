from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .tactile_filter import TactileDenoiseConfig, as_distributed_force_array


try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on local runtime env
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


_HELD_DEVICES = []


@dataclass
class ForceVizConfig:
    stride: int = 12
    scale: int = 2
    arrow_scale: float = 18.0
    min_arrow_norm: float = 0.015
    deadband_abs: float = 5e-5
    temporal_alpha: float = 0.85
    spatial_blur_ksize: int = 5
    arrow_pool: int = 16
    norm_percentile: float = 99.0
    norm_ema_alpha: float = 0.2
    fixed_ref: Optional[float] = None
    heatmap_alpha: float = 0.65
    colorbar_width: int = 42
    show_stats: bool = True
    show_mode_label: bool = True
    mode: str = "absolute"
    baseline_frames: int = 8


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for visualization. Install external dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from _CV2_IMPORT_ERROR
    return cv2


def _as_distributed_force_array(distributed_force: Any) -> np.ndarray:
    return as_distributed_force_array(distributed_force)


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    value = float(np.percentile(finite, percentile))
    return max(value, 1e-6)


def _colormap() -> int:
    cv = require_cv2()
    return getattr(cv, "COLORMAP_TURBO", cv.COLORMAP_JET)


def _put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.48) -> None:
    cv = require_cv2()
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv.LINE_AA)


def _make_colorbar(height: int, width: int, ref: float) -> np.ndarray:
    cv = require_cv2()
    ramp = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    bar = np.repeat(ramp, max(1, width), axis=1)
    out = cv.applyColorMap(bar, _colormap())
    _put_text(out, f"{ref:.3g}", (3, 16), 0.38)
    _put_text(out, "0", (3, height - 7), 0.38)
    return out


class DistributedForceXYVisualizer:
    """Render DM-Tac getDistributeForce() XY vectors as colored arrows."""

    def __init__(self, config: Optional[ForceVizConfig] = None) -> None:
        self.config = config or ForceVizConfig()
        self._norm_ref: Dict[Any, float] = {}
        self._prev_force: Dict[Any, np.ndarray] = {}
        self._prev_display_force: Dict[Any, np.ndarray] = {}
        self._filtered_force: Dict[Any, np.ndarray] = {}
        self._baseline_sum: Dict[Any, np.ndarray] = {}
        self._baseline_count: Dict[Any, int] = {}
        self._baseline: Dict[Any, np.ndarray] = {}

    @classmethod
    def display_only(cls, config: Optional[ForceVizConfig] = None) -> "DistributedForceXYVisualizer":
        source = config or ForceVizConfig()
        display_config = ForceVizConfig(
            stride=source.stride,
            scale=source.scale,
            arrow_scale=source.arrow_scale,
            min_arrow_norm=source.min_arrow_norm,
            deadband_abs=0.0,
            temporal_alpha=1.0,
            spatial_blur_ksize=0,
            arrow_pool=source.arrow_pool,
            norm_percentile=source.norm_percentile,
            norm_ema_alpha=source.norm_ema_alpha,
            fixed_ref=source.fixed_ref,
            heatmap_alpha=source.heatmap_alpha,
            colorbar_width=source.colorbar_width,
            show_stats=source.show_stats,
            show_mode_label=False,
            mode="absolute",
            baseline_frames=1,
        )
        return cls(display_config)

    def render_force(
        self,
        distributed_force: Any,
        *,
        key: Any = "default",
        title: Optional[str] = None,
    ) -> np.ndarray:
        cv = require_cv2()
        cfg = self.config
        raw_force = _as_distributed_force_array(distributed_force)
        force, mode_label = self._display_force(raw_force, key)
        force = self._filter_display_force(force, key)
        xy = np.nan_to_num(force[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
        mag = np.linalg.norm(xy, axis=2)
        h, w = mag.shape
        stats = distributed_force_stats(force, self._prev_display_force.get(key))
        self._prev_display_force[key] = force.copy()
        self._prev_force[key] = raw_force.copy()

        if cfg.fixed_ref is not None and cfg.fixed_ref > 0:
            ref = float(cfg.fixed_ref)
        else:
            target_ref = _safe_percentile(mag, cfg.norm_percentile)
            prev_ref = self._norm_ref.get(key)
            if prev_ref is None:
                ref = target_ref
            else:
                alpha = min(max(float(cfg.norm_ema_alpha), 0.0), 1.0)
                ref = (1.0 - alpha) * prev_ref + alpha * target_ref
                ref = max(ref, 1e-6)
            self._norm_ref[key] = ref

        norm = np.clip(mag / ref, 0.0, 1.0)
        heat_u8 = (norm * 255.0).astype(np.uint8)
        heat = cv.applyColorMap(heat_u8, _colormap())
        base = np.zeros_like(heat)
        image = cv.addWeighted(heat, cfg.heatmap_alpha, base, 1.0 - cfg.heatmap_alpha, 0.0)

        scale = max(1, int(cfg.scale))
        if scale != 1:
            image = cv.resize(image, (w * scale, h * scale), interpolation=cv.INTER_LINEAR)

        stride = max(1, int(cfg.stride))
        max_arrow_px = max(1.0, float(cfg.arrow_scale) * scale)
        min_arrow_norm = max(0.0, float(cfg.min_arrow_norm))
        color_lut = cv.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), _colormap()).reshape(256, 3)

        y0 = stride // 2
        x0 = stride // 2
        for y in range(y0, h, stride):
            for x in range(x0, w, stride):
                vx, vy = self._arrow_vector(xy, x, y)
                length = float(mag[y, x])
                if cfg.arrow_pool and cfg.arrow_pool > 1:
                    length = float(np.hypot(vx, vy))
                if length <= 1e-8:
                    continue
                n = min(length / ref, 1.0)
                if n < min_arrow_norm:
                    continue

                arrow_len = min(max_arrow_px, max(1.0, n * max_arrow_px))
                dx = vx / length * arrow_len
                dy = vy / length * arrow_len
                start = (int(round(x * scale)), int(round(y * scale)))
                end = (int(round(x * scale + dx)), int(round(y * scale + dy)))
                color = tuple(int(c) for c in color_lut[int(round(n * 255.0))].tolist())

                cv.arrowedLine(image, start, end, (0, 0, 0), 3, cv.LINE_AA, tipLength=0.35)
                cv.arrowedLine(image, start, end, color, 1, cv.LINE_AA, tipLength=0.35)

        shown_title = str(title) if title else ""
        if mode_label and cfg.show_mode_label:
            shown_title = f"{shown_title} {mode_label}".strip()
        if shown_title:
            _put_text(image, shown_title, (8, 20), 0.55)
        if cfg.show_stats:
            line = (
                f"XY mean={stats['xy_mean']:.3g} max={stats['xy_max']:.3g} "
                f"dXY max={stats.get('dxy_max', 0.0):.3g}"
            )
            _put_text(image, line, (8, image.shape[0] - 10), 0.48)

        if cfg.colorbar_width > 0:
            colorbar = _make_colorbar(image.shape[0], int(cfg.colorbar_width), ref)
            image = np.concatenate([image, colorbar], axis=1)

        return image

    def _filter_display_force(self, force: np.ndarray, key: Any) -> np.ndarray:
        cv = require_cv2()
        cfg = self.config
        out = force.astype(np.float32, copy=True)

        blur_ksize = int(cfg.spatial_blur_ksize)
        if blur_ksize > 1:
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            out[..., :2] = cv.GaussianBlur(out[..., :2], (blur_ksize, blur_ksize), 0)

        alpha = float(cfg.temporal_alpha)
        if 0.0 < alpha < 1.0:
            prev = self._filtered_force.get(key)
            if prev is not None and prev.shape == out.shape:
                out = prev * (1.0 - alpha) + out * alpha
        self._filtered_force[key] = out.copy()

        deadband = max(0.0, float(cfg.deadband_abs))
        if deadband > 0.0:
            mag = np.linalg.norm(out[..., :2], axis=2)
            xy = out[..., :2]
            xy[mag < deadband] = 0.0
            out[..., :2] = xy

        return out

    def _arrow_vector(self, xy: np.ndarray, x: int, y: int) -> Tuple[float, float]:
        pool = int(self.config.arrow_pool)
        if pool <= 1:
            return float(xy[y, x, 0]), float(xy[y, x, 1])

        half = max(1, pool // 2)
        y1 = max(0, y - half)
        y2 = min(xy.shape[0], y + half + 1)
        x1 = max(0, x - half)
        x2 = min(xy.shape[1], x + half + 1)
        patch = xy[y1:y2, x1:x2]
        vec = np.nanmean(patch, axis=(0, 1))
        return float(vec[0]), float(vec[1])

    def _display_force(self, raw_force: np.ndarray, key: Any) -> Tuple[np.ndarray, str]:
        mode = str(self.config.mode).strip().lower()
        if mode == "absolute":
            return raw_force, "absolute"

        if mode == "frame_delta":
            prev = self._prev_force.get(key)
            if prev is None or prev.shape != raw_force.shape:
                return np.zeros_like(raw_force), "frame_delta waiting"
            return raw_force - prev, "frame_delta"

        if mode != "delta":
            raise ValueError("ForceVizConfig.mode must be 'delta', 'absolute', or 'frame_delta'")

        baseline = self._baseline.get(key)
        if baseline is not None and baseline.shape == raw_force.shape:
            return raw_force - baseline, "delta"

        need = max(1, int(self.config.baseline_frames))
        if key not in self._baseline_sum or self._baseline_sum[key].shape != raw_force.shape:
            self._baseline_sum[key] = np.zeros_like(raw_force, dtype=np.float64)
            self._baseline_count[key] = 0
        self._baseline_sum[key] += raw_force.astype(np.float64, copy=False)
        self._baseline_count[key] += 1
        count = self._baseline_count[key]
        if count >= need:
            self._baseline[key] = (self._baseline_sum[key] / float(count)).astype(np.float32)
            return raw_force - self._baseline[key], "delta"
        return np.zeros_like(raw_force), f"baseline {count}/{need}"

    def render_frames(self, frames: Mapping[Any, Any]) -> np.ndarray:
        panels = []
        for dev_id, frame in frames.items():
            if frame is None or getattr(frame, "distributed_force", None) is None:
                panels.append(self.render_blank(title=f"dev {dev_id}: no frame"))
                continue
            fid = getattr(frame, "fid", None)
            title = f"dev {dev_id}"
            if fid is not None:
                title += f" fid={fid}"
            panels.append(self.render_force(frame.distributed_force, key=dev_id, title=title))
        if not panels:
            return self.render_blank(title="no tactile frame")
        return _hstack_equal_height(panels)

    def render_blank(self, title: str = "no data", shape: Tuple[int, int] = (288, 384)) -> np.ndarray:
        cv = require_cv2()
        cfg = self.config
        h, w = shape
        scale = max(1, int(cfg.scale))
        image = np.zeros((h * scale, w * scale + max(0, cfg.colorbar_width), 3), dtype=np.uint8)
        _put_text(image, title, (8, 24), 0.55)
        return image


def _hstack_equal_height(images: Sequence[np.ndarray]) -> np.ndarray:
    cv = require_cv2()
    max_h = max(img.shape[0] for img in images)
    resized = []
    for img in images:
        if img.shape[0] == max_h:
            resized.append(img)
            continue
        new_w = int(round(img.shape[1] * max_h / img.shape[0]))
        resized.append(cv.resize(img, (new_w, max_h), interpolation=cv.INTER_LINEAR))
    return np.concatenate(resized, axis=1)


def distributed_force_stats(distributed_force: Any, prev_force: Any = None) -> Dict[str, Any]:
    force = _as_distributed_force_array(distributed_force)
    xy = np.nan_to_num(force[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
    xy_mag = np.linalg.norm(xy, axis=2)
    out: Dict[str, Any] = {
        "shape": tuple(force.shape),
        "dtype": str(force.dtype),
        "xy_mean": float(np.nanmean(xy_mag)) if xy_mag.size else 0.0,
        "xy_max": float(np.nanmax(xy_mag)) if xy_mag.size else 0.0,
        "xy_p99": _safe_percentile(xy_mag, 99.0),
    }
    if force.shape[2] >= 3:
        xz = np.nan_to_num(force[..., (0, 2)], nan=0.0, posinf=0.0, neginf=0.0)
        xz_mag = np.linalg.norm(xz, axis=2)
        y_abs = np.abs(np.nan_to_num(force[..., 1], nan=0.0, posinf=0.0, neginf=0.0))
        z_abs = np.abs(np.nan_to_num(force[..., 2], nan=0.0, posinf=0.0, neginf=0.0))
        out["xz_mean"] = float(np.nanmean(xz_mag)) if xz_mag.size else 0.0
        out["xz_max"] = float(np.nanmax(xz_mag)) if xz_mag.size else 0.0
        out["xz_p99"] = _safe_percentile(xz_mag, 99.0)
        out["y_abs_mean"] = float(np.nanmean(y_abs)) if y_abs.size else 0.0
        out["y_abs_max"] = float(np.nanmax(y_abs)) if y_abs.size else 0.0
        out["z_abs_mean"] = float(np.nanmean(z_abs)) if z_abs.size else 0.0
        out["z_abs_max"] = float(np.nanmax(z_abs)) if z_abs.size else 0.0
        xyz_mag = np.linalg.norm(np.nan_to_num(force[..., :3], nan=0.0, posinf=0.0, neginf=0.0), axis=2)
        out["xyz_mean"] = float(np.nanmean(xyz_mag)) if xyz_mag.size else 0.0
        out["xyz_max"] = float(np.nanmax(xyz_mag)) if xyz_mag.size else 0.0

    if prev_force is not None:
        prev = _as_distributed_force_array(prev_force)
        if prev.shape == force.shape:
            delta = np.nan_to_num(force - prev, nan=0.0, posinf=0.0, neginf=0.0)
            dxy = np.linalg.norm(delta[..., :2], axis=2)
            out["dxy_mean"] = float(np.nanmean(dxy)) if dxy.size else 0.0
            out["dxy_max"] = float(np.nanmax(dxy)) if dxy.size else 0.0
            if delta.shape[2] >= 3:
                dxz = np.linalg.norm(delta[..., (0, 2)], axis=2)
                out["dxz_mean"] = float(np.nanmean(dxz)) if dxz.size else 0.0
                out["dxz_max"] = float(np.nanmax(dxz)) if dxz.size else 0.0
                dy_abs = np.abs(delta[..., 1])
                out["dy_abs_mean"] = float(np.nanmean(dy_abs)) if dy_abs.size else 0.0
                out["dy_abs_max"] = float(np.nanmax(dy_abs)) if dy_abs.size else 0.0
            dxyz = np.linalg.norm(delta[..., : min(3, delta.shape[2])], axis=2)
            out["dxyz_mean"] = float(np.nanmean(dxyz)) if dxyz.size else 0.0
            out["dxyz_max"] = float(np.nanmax(dxyz)) if dxyz.size else 0.0
    return out


def format_distributed_force_stats(dev_id: Any, fid: Any, stats: Mapping[str, Any]) -> str:
    parts = [
        f"dev={dev_id}",
        f"fid={fid}",
        f"xy_mean={stats['xy_mean']:.6g}",
        f"xy_max={stats['xy_max']:.6g}",
        f"dxy_max={stats.get('dxy_max', 0.0):.6g}",
    ]
    if "z_abs_max" in stats:
        parts.extend(
            [
                f"xz_max={stats.get('xz_max', 0.0):.6g}",
                f"dxz_max={stats.get('dxz_max', 0.0):.6g}",
                f"y_abs_mean={stats.get('y_abs_mean', 0.0):.6g}",
                f"y_abs_max={stats.get('y_abs_max', 0.0):.6g}",
                f"dy_abs_max={stats.get('dy_abs_max', 0.0):.6g}",
                f"z_abs_mean={stats['z_abs_mean']:.6g}",
                f"z_abs_max={stats['z_abs_max']:.6g}",
                f"dxyz_max={stats.get('dxyz_max', 0.0):.6g}",
            ]
        )
    return " ".join(parts)


def _format_timing_ms(timings: Mapping[str, Any]) -> str:
    order = (
        "getDevStatus",
        "wait_for_new",
        "distributed_force",
        "force",
        "shear",
        "depth",
        "deformation2d",
        "raw",
        "infer",
        "events",
    )
    parts = []
    for key in order:
        if key in timings:
            parts.append(f"{key}={float(timings[key]):.1f}ms")
    for key, value in timings.items():
        if key not in order:
            parts.append(f"{key}={float(value):.1f}ms")
    return " ".join(parts)


def probe_distributed_force(
    *,
    tactile_config,
    timeout_ms: int = 100,
    count: int = 30,
    period_s: float = 0.0,
    reset_tactile: bool = False,
    warmup_frames: int = 0,
    close_devices: bool = True,
) -> None:
    from .device import DMTacGripperDevice

    tactile_config.fields = ("distributed_force",)
    prev_by_dev: Dict[Any, np.ndarray] = {}
    dev = DMTacGripperDevice(
        tactile=tactile_config,
        connect_gripper=False,
        connect_tactile=True,
    ).connect()
    try:
        _prepare_tactile(dev, timeout_ms=timeout_ms, reset_tactile=reset_tactile, warmup_frames=warmup_frames)
        for i in range(count):
            read_started_at = time.perf_counter()
            frames = dev.read_tactile(timeout_ms=timeout_ms)
            read_ms = (time.perf_counter() - read_started_at) * 1000.0
            print(
                f"frame {i} read={read_ms:.1f}ms "
                f"profile={getattr(tactile_config, 'compute_profile', '')} "
                f"parallel={bool(getattr(tactile_config, 'parallel_read', False))}"
            )
            for dev_id, frame in frames.items():
                if frame is None or getattr(frame, "distributed_force", None) is None:
                    print(f"  dev={dev_id} no distributed_force frame")
                    continue
                force = frame.distributed_force
                stats = distributed_force_stats(force, prev_by_dev.get(dev_id))
                prev_by_dev[dev_id] = _as_distributed_force_array(force).copy()
                line = "  " + format_distributed_force_stats(dev_id, getattr(frame, "fid", None), stats)
                timings = getattr(frame, "timings_ms", {}) or {}
                if timings:
                    line += " timings[" + _format_timing_ms(timings) + "]"
                errors = getattr(frame, "errors", {}) or {}
                if errors:
                    line += f" errors={dict(errors)}"
                print(line)
            if period_s > 0:
                time.sleep(period_s)
    finally:
        if close_devices:
            dev.close()
        else:
            _HELD_DEVICES.append(dev)


def run_live_distributed_force_xy(
    *,
    tactile_config,
    timeout_ms: int = 100,
    count: int = 0,
    max_hz: float = 0.0,
    window: str = "DM-Tac distributed_force XY",
    viz_config: Optional[ForceVizConfig] = None,
    reset_tactile: bool = False,
    warmup_frames: int = 0,
    close_devices: bool = True,
) -> None:
    cv = require_cv2()

    from .device import DMTacGripperDevice

    tactile_config.fields = ("distributed_force",)
    if viz_config is not None and str(viz_config.mode).strip().lower() != "absolute":
        tactile_config.denoise_distributed_force = True
        tactile_config.denoise_config = TactileDenoiseConfig(
            enabled=True,
            mode=viz_config.mode,
            baseline_frames=viz_config.baseline_frames,
            deadband_abs=viz_config.deadband_abs,
            temporal_alpha=viz_config.temporal_alpha,
            spatial_blur_ksize=viz_config.spatial_blur_ksize,
        )
    visualizer = DistributedForceXYVisualizer.display_only(viz_config)
    frame_count = 0
    fps_start = time.perf_counter()
    fps_value = 0.0

    dev = DMTacGripperDevice(
        tactile=tactile_config,
        connect_gripper=False,
        connect_tactile=True,
    ).connect()
    try:
        _prepare_tactile(dev, timeout_ms=timeout_ms, reset_tactile=reset_tactile, warmup_frames=warmup_frames)
        try:
            while True:
                loop_start = time.perf_counter()
                frames = dev.read_tactile(timeout_ms=timeout_ms)
                image = visualizer.render_frames(frames)

                frame_count += 1
                elapsed = time.perf_counter() - fps_start
                if elapsed >= 0.5:
                    fps_value = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.perf_counter()
                _put_text(image, f"fps={fps_value:.1f}  q/ESC quit", (8, 42), 0.5)

                cv.imshow(window, image)
                key = cv.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if count > 0:
                    count -= 1
                    if count <= 0:
                        break
                if max_hz > 0:
                    target_dt = 1.0 / float(max_hz)
                    remain = target_dt - (time.perf_counter() - loop_start)
                    if remain > 0:
                        time.sleep(remain)
        finally:
            cv.destroyWindow(window)
    finally:
        if close_devices:
            dev.close()
        else:
            _HELD_DEVICES.append(dev)


def _prepare_tactile(dev, *, timeout_ms: int, reset_tactile: bool, warmup_frames: int) -> None:
    print("tactile ready:", dev.wait_tactile_ready(timeout_s=10))
    if reset_tactile:
        print("reset tactile: keep all sensor surfaces unloaded")
        dev.reset_tactile()
        print("tactile ready after reset:", dev.wait_tactile_ready(timeout_s=10))

    frames = max(0, int(warmup_frames))
    if frames <= 0:
        return
    if getattr(dev.tactile_config, "denoise_distributed_force", False):
        print(f"warmup: dropping {frames} raw frames, then rebuilding denoise baseline")
    else:
        print(f"warmup: dropping {frames} raw frames")
    dev.warmup_tactile_filter(frames=frames, timeout_ms=timeout_ms)

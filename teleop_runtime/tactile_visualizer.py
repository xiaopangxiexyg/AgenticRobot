from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .backends.hands import InspireTacHandBackend


@dataclass(frozen=True)
class TactileVisualizerConfig:
    enabled_sides: tuple[str, ...] = ("left", "right")
    window_name: str = "inspire_hand_tactile_uv"
    image_size: tuple[int, int] = (640, 640)
    resizable_window: bool = True
    value_scale: float = 0.05
    point_radius_min: int = 1
    point_radius_max: int = 6
    panel_gap_px: int = 24
    panel_padding_px: int = 20
    layout_margin_px: int = 36
    ring_radius_px: int = 2
    ring_thickness_px: int = 1
    ring_color: tuple[int, int, int] = (95, 95, 95)
    low_response_floor: float = 0.01
    color_value_cap: float = 1.0
    radius_value_cap: float = 1.5
    response_gamma: float = 0.55

    def __post_init__(self) -> None:
        if self.value_scale <= 0.0:
            raise ValueError("value_scale must be positive")
        if self.point_radius_min <= 0 or self.point_radius_max <= 0:
            raise ValueError("point radii must be positive")
        if self.point_radius_min > self.point_radius_max:
            raise ValueError("point_radius_min cannot exceed point_radius_max")


class TactileUVVisualizer:
    _REGION_NAMES: tuple[str, ...] = (
        "little_1",
        "little_2",
        "little_3",
        "ring_1",
        "ring_2",
        "ring_3",
        "middle_1",
        "middle_2",
        "middle_3",
        "index_1",
        "index_2",
        "index_3",
        "thumb_1",
        "thumb_2",
        "thumb_3",
        "thumb_4",
        "palm",
    )
    _REGION_SIZES: tuple[tuple[int, int], ...] = (
        (3, 3),
        (12, 8),
        (10, 8),
        (3, 3),
        (12, 8),
        (10, 8),
        (3, 3),
        (12, 8),
        (10, 8),
        (3, 3),
        (12, 8),
        (10, 8),
        (3, 3),
        (12, 8),
        (3, 3),
        (12, 8),
        (8, 14),
    )
    _RIGHT_REGION_LAYOUT: dict[str, tuple[float, float, float, float]] = {
        "little_1": (0.22, 0.14, 0.08, 0.06),
        "little_2": (0.22, 0.25, 0.10, 0.15),
        "little_3": (0.22, 0.40, 0.10, 0.14),
        "ring_1": (0.36, 0.12, 0.08, 0.06),
        "ring_2": (0.36, 0.23, 0.10, 0.15),
        "ring_3": (0.36, 0.38, 0.10, 0.14),
        "middle_1": (0.50, 0.10, 0.08, 0.06),
        "middle_2": (0.50, 0.21, 0.10, 0.16),
        "middle_3": (0.50, 0.36, 0.10, 0.14),
        "index_1": (0.64, 0.12, 0.08, 0.06),
        "index_2": (0.64, 0.23, 0.10, 0.15),
        "index_3": (0.64, 0.38, 0.10, 0.14),
        "thumb_1": (0.80, 0.26, 0.08, 0.06),
        "thumb_2": (0.78, 0.38, 0.10, 0.15),
        "thumb_3": (0.74, 0.50, 0.08, 0.06),
        "thumb_4": (0.70, 0.62, 0.10, 0.15),
        "palm": (0.50, 0.74, 0.58, 0.22),
    }

    def __init__(self, config: TactileVisualizerConfig):
        self.config = config
        self._cv2 = self._import_cv2()
        self._panel_background = self._build_panel_background()
        self._layout_pixel_coords = {
            "left": self._build_layout_pixel_coords("left"),
            "right": self._build_layout_pixel_coords("right"),
        }
        self._canvas = self._build_canvas_template()
        self._latest_canvas: np.ndarray | None = None
        self._window_created = False
        self._closed = False

    @staticmethod
    def _import_cv2() -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV (cv2) is required for tactile visualization") from exc
        return cv2

    def _build_panel_background(self) -> np.ndarray:
        width, height = self.config.image_size
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        bg[...] = np.array([22, 22, 22], dtype=np.uint8)
        return bg

    def _build_canvas_template(self) -> np.ndarray:
        panel_w, panel_h = self.config.image_size
        pad = int(self.config.panel_padding_px)
        gap = int(self.config.panel_gap_px)
        canvas_w = panel_w * 2 + gap + pad * 2
        canvas_h = panel_h + pad * 2 + 36
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[...] = np.array([10, 10, 10], dtype=np.uint8)
        return canvas

    def _build_layout_pixel_coords(self, side: str) -> np.ndarray:
        width, height = self.config.image_size
        margin = int(self.config.layout_margin_px)
        draw_w = max(1, width - 2 * margin)
        draw_h = max(1, height - 2 * margin)
        coords = np.zeros((1062, 2), dtype=np.float32)
        cursor = 0
        for region_name, (rows, cols) in zip(self._REGION_NAMES, self._REGION_SIZES):
            cx, cy, rw, rh = self._region_layout(side=side, region_name=region_name)
            xs = np.linspace(cx - rw * 0.5, cx + rw * 0.5, cols, dtype=np.float32)
            ys = np.linspace(cy - rh * 0.5, cy + rh * 0.5, rows, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
            count = rows * cols
            coords[cursor : cursor + count] = points
            cursor += count
        x = np.clip(coords[:, 0], 0.0, 1.0)
        y = np.clip(coords[:, 1], 0.0, 1.0)
        px = margin + x * (draw_w - 1)
        py = margin + y * (draw_h - 1)
        return np.rint(np.stack([px, py], axis=1)).astype(np.int32)

    def _region_layout(self, *, side: str, region_name: str) -> tuple[float, float, float, float]:
        cx, cy, rw, rh = self._RIGHT_REGION_LAYOUT[region_name]
        if side == "left":
            cx = 1.0 - cx
        return cx, cy, rw, rh

    def _draw_panel(self, *, side: str, tactile_1062: np.ndarray | None) -> np.ndarray:
        panel = self._panel_background.copy()
        points = self._layout_pixel_coords[side]
        for center_xy in points:
            self._cv2.circle(
                panel,
                (int(center_xy[0]), int(center_xy[1])),
                int(self.config.ring_radius_px),
                tuple(int(v) for v in self.config.ring_color),
                int(self.config.ring_thickness_px),
                lineType=self._cv2.LINE_AA,
            )
        if tactile_1062 is None:
            self._cv2.putText(
                panel,
                "no tactile",
                (12, 24),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
                self._cv2.LINE_AA,
            )
            return panel
        values = np.asarray(tactile_1062, dtype=np.float32).reshape(-1)
        if values.shape[0] != 1062:
            self._cv2.putText(
                panel,
                "invalid tactile shape",
                (12, 24),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (60, 90, 240),
                2,
                self._cv2.LINE_AA,
            )
            return panel
        valid_mask = np.isfinite(values) & (values >= 0.0)
        if not np.any(valid_mask):
            return panel
        color_strength, size_strength = self._map_visual_strength(values)
        scaled_uint8 = np.rint(color_strength * 255.0).astype(np.uint8)
        colors = self._cv2.applyColorMap(scaled_uint8.reshape(-1, 1), self._cv2.COLORMAP_TURBO)
        colors = colors.reshape(-1, 3)
        r_min = int(self.config.point_radius_min)
        r_max = int(self.config.point_radius_max)
        for idx in np.flatnonzero(valid_mask):
            if float(color_strength[idx]) <= 0.0 and float(size_strength[idx]) <= 0.0:
                continue
            radius = int(round(r_min + float(size_strength[idx]) * (r_max - r_min)))
            center = (int(points[idx, 0]), int(points[idx, 1]))
            color = tuple(int(v) for v in colors[idx].tolist())
            self._cv2.circle(panel, center, radius, color, -1, lineType=self._cv2.LINE_AA)
        return panel

    def _map_visual_strength(self, tactile_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positive = np.maximum(np.asarray(tactile_vec, dtype=np.float32), 0.0)
        floor = float(self.config.low_response_floor)
        shifted = np.maximum(positive - floor, 0.0)
        color_cap = max(float(self.config.color_value_cap), float(self.config.value_scale) * 10.0)
        color_strength = np.log1p(shifted / floor) / np.log1p(color_cap / floor)
        color_strength = np.clip(color_strength, 0.0, 1.0)
        color_strength = np.power(color_strength, float(self.config.response_gamma), dtype=np.float32)
        radius_cap = max(float(self.config.radius_value_cap), float(self.config.value_scale) * 20.0)
        size_strength = np.log1p(shifted / floor) / np.log1p(radius_cap / floor)
        size_strength = np.clip(size_strength, 0.0, 1.0)
        return color_strength, size_strength

    def _ensure_window(self, canvas: np.ndarray) -> None:
        if self._window_created:
            return
        flag = self._cv2.WINDOW_AUTOSIZE
        if self.config.resizable_window:
            flag = self._cv2.WINDOW_NORMAL | int(
                getattr(self._cv2, "WINDOW_FREERATIO", 0)
            )
        self._cv2.namedWindow(self.config.window_name, flag)
        if self.config.resizable_window:
            set_property = getattr(self._cv2, "setWindowProperty", None)
            aspect_property = getattr(self._cv2, "WND_PROP_ASPECT_RATIO", None)
            free_ratio = getattr(self._cv2, "WINDOW_FREERATIO", None)
            if (
                callable(set_property)
                and aspect_property is not None
                and free_ratio is not None
            ):
                try:
                    set_property(self.config.window_name, aspect_property, free_ratio)
                except Exception:
                    pass
            height, width = canvas.shape[:2]
            self._cv2.resizeWindow(self.config.window_name, int(width), int(height))
        self._window_created = True

    def update(self, tactile_by_side: dict[str, np.ndarray | None]) -> None:
        if self._closed:
            return
        canvas = self._canvas.copy()
        panel_w, panel_h = self.config.image_size
        pad = int(self.config.panel_padding_px)
        gap = int(self.config.panel_gap_px)
        top = pad + 28
        left_x = pad
        right_x = pad + panel_w + gap
        for side, x0 in (("left", left_x), ("right", right_x)):
            panel = self._draw_panel(side=side, tactile_1062=tactile_by_side.get(side))
            canvas[top : top + panel_h, x0 : x0 + panel_w] = panel
            self._cv2.putText(
                canvas,
                side,
                (x0, top - 8),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (210, 210, 210),
                2,
                self._cv2.LINE_AA,
            )
        self._ensure_window(canvas)
        self._latest_canvas = canvas
        self._cv2.imshow(self.config.window_name, canvas)
        self._cv2.waitKey(1)

    def poll_events(self) -> None:
        if self._closed:
            return
        if self._latest_canvas is not None:
            self._cv2.imshow(self.config.window_name, self._latest_canvas)
        self._cv2.waitKey(1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cv2.destroyWindow(self.config.window_name)
            self._cv2.waitKey(1)
        except Exception:
            pass


_ASYNC_CLOSE_SENTINEL = "__teleop_tactile_visualizer_close__"


def _run_tactile_visualizer_process(
    config: TactileVisualizerConfig,
    input_queue: Any,
    ready_queue: Any,
) -> None:
    visualizer: TactileUVVisualizer | None = None
    try:
        visualizer = TactileUVVisualizer(config)
        visualizer.update({})
    except Exception as exc:  # noqa: BLE001
        ready_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return
    ready_queue.put({"ok": True})
    try:
        while True:
            try:
                payload = input_queue.get(timeout=0.015)
            except queue.Empty:
                visualizer.poll_events()
                continue
            if payload == _ASYNC_CLOSE_SENTINEL:
                break
            if isinstance(payload, dict):
                visualizer.update(payload)
            else:
                visualizer.poll_events()
    except KeyboardInterrupt:
        pass
    finally:
        if visualizer is not None:
            visualizer.close()


class AsyncTactileUVVisualizer:
    """Run the OpenCV tactile window in a fresh process.

    SAPIEN viewers and OpenCV HighGUI can load different GUI/render backends in
    one process. A spawn child keeps the tactile window independent of SAPIEN.
    """

    def __init__(
        self,
        config: TactileVisualizerConfig,
        *,
        startup_timeout_sec: float = 3.0,
    ):
        self.config = config
        self._closed = False
        self._ctx = mp.get_context("spawn")
        self._input_queue = self._ctx.Queue(maxsize=2)
        self._ready_queue = self._ctx.Queue(maxsize=1)
        self._process = self._ctx.Process(
            target=_run_tactile_visualizer_process,
            args=(self.config, self._input_queue, self._ready_queue),
            name="tactile_uv_visualizer",
            daemon=True,
        )
        self._process.start()
        try:
            ready = self._ready_queue.get(timeout=float(startup_timeout_sec))
        except queue.Empty:
            if self._process.is_alive():
                return
            raise RuntimeError("Tactile visualizer process exited during startup.")
        if not isinstance(ready, dict) or ready.get("ok") is not True:
            error = ready.get("error") if isinstance(ready, dict) else repr(ready)
            details = ready.get("traceback", "") if isinstance(ready, dict) else ""
            self.close()
            raise RuntimeError(
                "Tactile visualizer process failed to start: "
                f"{error}\n{details}".rstrip()
            )

    def update(self, tactile_by_side: dict[str, np.ndarray | None]) -> None:
        if self._closed or not self._process.is_alive():
            return
        payload = {
            str(side): None if tactile is None else np.asarray(tactile, dtype=np.float32).reshape(-1)
            for side, tactile in tactile_by_side.items()
        }
        try:
            self._input_queue.put_nowait(payload)
            return
        except queue.Full:
            pass
        try:
            self._input_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._input_queue.put_nowait(payload)
        except queue.Full:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            try:
                self._input_queue.put_nowait(_ASYNC_CLOSE_SENTINEL)
            except queue.Full:
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._input_queue.put_nowait(_ASYNC_CLOSE_SENTINEL)
                except queue.Full:
                    pass
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        else:
            self._process.join(timeout=0.1)
        for q in (self._input_queue, self._ready_queue):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass


@dataclass(frozen=True)
class DMTacForceVisualizerConfig:
    enabled_sides: tuple[str, ...] = ("left", "right")
    window_name: str = "dmtac_gripper_tactile"
    dmtac_root: str | Path | None = None
    grid_rows: int = 20
    grid_cols: int = 30
    cell_size: int = 18
    vector_stride: int = 2
    stride: int = 16
    scale: int = 1
    arrow_scale: float = 0.42
    min_arrow_norm: float = 0.02
    # Use one physical scale across every sensor and frame. Per-frame robust
    # normalization makes arbitrarily small startup noise appear fully
    # saturated and prevents visual comparison between sensors.
    fixed_ref: float | None = 1.0
    normal_sign: float = 1.0
    mode: str = "absolute"
    resizable_window: bool = True


class DMTacDistributedForceVisualizer:
    """Render DMTAC distributed_force as 20x30x3 pressure/vector grids."""

    def __init__(self, config: DMTacForceVisualizerConfig):
        self.config = config
        self._cv2 = None
        self._window_created = False
        self._latest_canvas: np.ndarray | None = None
        self._closed = False

    def _ensure_cv2(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError("OpenCV (cv2) is required for DMTAC tactile visualization.") from exc
            self._cv2 = cv2
        return self._cv2

    @staticmethod
    def _coarse_force_grid(
        distributed_force: Any,
        *,
        rows: int,
        cols: int,
        cv2: Any,
    ) -> np.ndarray:
        force = np.asarray(distributed_force, dtype=np.float32)
        if force.ndim != 3 or force.shape[2] < 3:
            raise ValueError("distributed_force must have shape (H, W, C) with at least three channels")
        row_count = max(1, int(rows))
        col_count = max(1, int(cols))
        source = np.ascontiguousarray(
            np.nan_to_num(force[..., :3], nan=0.0, posinf=0.0, neginf=0.0)
        )
        resized = cv2.resize(
            source,
            (col_count, row_count),
            interpolation=cv2.INTER_AREA,
        )
        return np.nan_to_num(np.asarray(resized, dtype=np.float32), copy=False)

    @staticmethod
    def _robust_ref(values: np.ndarray, fixed_ref: float | None) -> float:
        if fixed_ref is not None and float(fixed_ref) > 0.0:
            return float(fixed_ref)
        finite = np.asarray(values, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size <= 0:
            return 1.0
        ref = float(np.percentile(np.abs(finite), 98))
        if ref <= 1e-8:
            ref = float(np.max(np.abs(finite))) if finite.size else 1.0
        return max(ref, 1e-6)

    def _render_blank(self, title: str = "no DMTAC tactile") -> np.ndarray:
        cv2 = self._ensure_cv2()
        rows = max(1, int(self.config.grid_rows))
        cols = max(1, int(self.config.grid_cols))
        cell = max(4, int(self.config.cell_size))
        image = np.zeros((rows * cell + 58, cols * cell, 3), dtype=np.uint8)
        cv2.putText(image, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(
            image,
            "20x30x3 force grid",
            (10, image.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return image

    def _render_force_grid(self, force: Any, *, title: str) -> np.ndarray:
        cv2 = self._ensure_cv2()
        rows = max(1, int(self.config.grid_rows))
        cols = max(1, int(self.config.grid_cols))
        cell = max(4, int(self.config.cell_size))
        grid = self._coarse_force_grid(force, rows=rows, cols=cols, cv2=cv2)
        pressure = np.maximum(float(self.config.normal_sign) * grid[..., 1], 0.0)
        pressure_ref = self._robust_ref(pressure, self.config.fixed_ref)
        pressure_u8 = np.clip(pressure / pressure_ref, 0.0, 1.0)
        heat = cv2.applyColorMap((pressure_u8 * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        image = cv2.resize(
            heat,
            (cols * cell, rows * cell),
            interpolation=cv2.INTER_NEAREST,
        )

        grid_color = (45, 45, 45)
        for col in range(cols + 1):
            x = col * cell
            cv2.line(image, (x, 0), (x, rows * cell), grid_color, 1)
        for row in range(rows + 1):
            y = row * cell
            cv2.line(image, (0, y), (cols * cell, y), grid_color, 1)

        plane = grid[..., (0, 2)]
        plane_norm = np.linalg.norm(plane, axis=2)
        vector_ref = self._robust_ref(plane_norm, self.config.fixed_ref)
        stride = max(1, int(self.config.vector_stride))
        max_len = float(cell) * max(0.05, float(self.config.arrow_scale))
        min_norm = float(self.config.min_arrow_norm)
        for row in range(0, rows, stride):
            for col in range(0, cols, stride):
                norm = float(plane_norm[row, col])
                if norm / vector_ref < min_norm:
                    continue
                direction = plane[row, col] / max(norm, 1e-8)
                length = min(max_len, max_len * norm / vector_ref)
                cx = int(round((col + 0.5) * cell))
                cy = int(round((row + 0.5) * cell))
                ex = int(round(cx + float(direction[0]) * length))
                ey = int(round(cy - float(direction[1]) * length))
                cv2.arrowedLine(
                    image,
                    (cx, cy),
                    (ex, ey),
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                    tipLength=0.35,
                )

        header_h = 36
        footer_h = 26
        canvas = np.zeros((header_h + image.shape[0] + footer_h, image.shape[1], 3), dtype=np.uint8)
        canvas[header_h : header_h + image.shape[0], :, :] = image
        cv2.putText(
            canvas,
            title,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        footer = (
            f"20x30x3  color=max(Fy,0) ref={pressure_ref:.3g}  "
            f"arrows=(Fx,Fz) ref={vector_ref:.3g}"
        )
        cv2.putText(
            canvas,
            footer,
            (8, canvas.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        return canvas

    @staticmethod
    def _hstack_equal_height(images: list[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("images must not be empty")
        max_h = max(int(img.shape[0]) for img in images)
        resized: list[np.ndarray] = []
        for img in images:
            if int(img.shape[0]) == max_h:
                resized.append(img)
                continue
            new_w = int(round(int(img.shape[1]) * max_h / max(1, int(img.shape[0]))))
            import cv2  # noqa: PLC0415

            resized.append(cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_LINEAR))
        return np.concatenate(resized, axis=1)

    @staticmethod
    def _vstack_equal_width(images: list[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("images must not be empty")
        max_w = max(int(img.shape[1]) for img in images)
        resized: list[np.ndarray] = []
        for img in images:
            if int(img.shape[1]) == max_w:
                resized.append(img)
                continue
            new_h = int(round(int(img.shape[0]) * max_w / max(1, int(img.shape[1]))))
            import cv2  # noqa: PLC0415

            resized.append(cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_LINEAR))
        return np.concatenate(resized, axis=0)

    def _ensure_window(self, canvas: np.ndarray) -> None:
        if self._window_created:
            return
        self._ensure_cv2()
        assert self._cv2 is not None
        flag = self._cv2.WINDOW_AUTOSIZE
        if self.config.resizable_window:
            flag = self._cv2.WINDOW_NORMAL | int(
                getattr(self._cv2, "WINDOW_FREERATIO", 0)
            )
        self._cv2.namedWindow(self.config.window_name, flag)
        if self.config.resizable_window:
            set_property = getattr(self._cv2, "setWindowProperty", None)
            aspect_property = getattr(self._cv2, "WND_PROP_ASPECT_RATIO", None)
            free_ratio = getattr(self._cv2, "WINDOW_FREERATIO", None)
            if (
                callable(set_property)
                and aspect_property is not None
                and free_ratio is not None
            ):
                try:
                    set_property(self.config.window_name, aspect_property, free_ratio)
                except Exception:
                    pass
            height, width = canvas.shape[:2]
            self._cv2.resizeWindow(self.config.window_name, int(width), int(height))
        self._window_created = True

    def update(
        self,
        tactile_by_side: dict[str, dict[str, np.ndarray] | None],
        *,
        health_by_side: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        if self._closed:
            return []
        side_panels: list[np.ndarray] = []
        for side in tuple(str(side) for side in self.config.enabled_sides):
            tactile_dict = tactile_by_side.get(side) or {}
            health = (health_by_side or {}).get(side) or {}
            status = str(health.get("status") or "ok")
            if status != "ok":
                tactile_dict = {}
            sensor_panels: list[np.ndarray] = []
            for sensor_key, force in sorted(tactile_dict.items(), key=lambda item: str(item[0])):
                sensor_panels.append(
                    self._render_force_grid(
                        force,
                        title=f"{side} dev {sensor_key}",
                    )
                )
            if not sensor_panels:
                detail = str(health.get("message") or status)
                sensor_panels.append(
                    self._render_blank(title=f"{side}: DMTAC {detail}")
                )
            side_panels.append(self._hstack_equal_height(sensor_panels))
        if not side_panels:
            side_panels.append(self._render_blank(title="no DMTAC tactile"))
        canvas = self._vstack_equal_width(side_panels)
        self._ensure_window(canvas)
        assert self._cv2 is not None
        self._latest_canvas = canvas
        self._cv2.imshow(self.config.window_name, canvas)
        key_code = self._cv2.waitKey(1)
        if key_code < 0:
            return []
        key = key_code & 0xFF
        if key == 3:
            raise KeyboardInterrupt
        if 0 < key < 128:
            return [chr(key).lower()]
        return []

    def poll_events(self) -> list[str]:
        if self._closed:
            return []
        if self._latest_canvas is None:
            return self.update({})
        assert self._cv2 is not None
        self._cv2.imshow(self.config.window_name, self._latest_canvas)
        key_code = self._cv2.waitKey(1)
        if key_code < 0:
            return []
        key = key_code & 0xFF
        if key == 3:
            raise KeyboardInterrupt
        if 0 < key < 128:
            return [chr(key).lower()]
        return []

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._cv2 is None:
            return
        try:
            self._cv2.destroyWindow(self.config.window_name)
            self._cv2.waitKey(1)
        except Exception:
            pass


_ASYNC_DMTAC_CLOSE_SENTINEL = "__teleop_dmtac_visualizer_close__"


def _run_dmtac_visualizer_process(
    config: DMTacForceVisualizerConfig,
    input_queue: Any,
    event_queue: Any,
    ready_queue: Any,
) -> None:
    visualizer: DMTacDistributedForceVisualizer | None = None
    try:
        visualizer = DMTacDistributedForceVisualizer(config)
        visualizer.update({})
    except Exception as exc:  # noqa: BLE001
        ready_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    ready_queue.put({"ok": True})
    try:
        while True:
            try:
                payload = input_queue.get(timeout=0.015)
            except queue.Empty:
                keys = visualizer.poll_events()
            else:
                if isinstance(payload, str) and payload == _ASYNC_DMTAC_CLOSE_SENTINEL:
                    break
                if not isinstance(payload, dict):
                    continue
                keys = visualizer.update(
                    payload.get("tactile") or {},
                    health_by_side=payload.get("health") or {},
                )
            for key in keys:
                try:
                    event_queue.put_nowait(str(key))
                except queue.Full:
                    break
    finally:
        visualizer.close()


class AsyncDMTacDistributedForceVisualizer:
    """Run the relatively expensive DMTAC grid renderer off the control loop."""

    def __init__(self, config: DMTacForceVisualizerConfig, *, startup_timeout_sec: float = 3.0):
        self.config = config
        self._closed = False
        self._ctx = mp.get_context("spawn")
        self._input_queue = self._ctx.Queue(maxsize=2)
        self._event_queue = self._ctx.Queue(maxsize=8)
        self._ready_queue = self._ctx.Queue(maxsize=1)
        self._process = self._ctx.Process(
            target=_run_dmtac_visualizer_process,
            args=(self.config, self._input_queue, self._event_queue, self._ready_queue),
            name="dmtac_force_visualizer",
            daemon=True,
        )
        self._process.start()
        try:
            ready = self._ready_queue.get(timeout=float(startup_timeout_sec))
        except queue.Empty:
            if self._process.is_alive():
                return
            raise RuntimeError("DMTAC visualizer process exited during startup.")
        if not isinstance(ready, dict) or ready.get("ok") is not True:
            error = ready.get("error") if isinstance(ready, dict) else repr(ready)
            self.close()
            raise RuntimeError(f"DMTAC visualizer process failed to start: {error}")

    def update(
        self,
        tactile_by_side: dict[str, dict[str, np.ndarray] | None],
        *,
        health_by_side: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        if self._closed or not self._process.is_alive():
            return []
        payload = {
            "tactile": {
                str(side): None
                if tactile is None
                else {
                    str(key): np.asarray(value, dtype=np.float32)
                    for key, value in tactile.items()
                }
                for side, tactile in tactile_by_side.items()
            },
            "health": dict(health_by_side or {}),
        }
        try:
            self._input_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait(payload)
            except queue.Full:
                pass
        keys: list[str] = []
        while True:
            try:
                keys.append(str(self._event_queue.get_nowait()))
            except queue.Empty:
                return keys

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            try:
                self._input_queue.put_nowait(_ASYNC_DMTAC_CLOSE_SENTINEL)
            except queue.Full:
                try:
                    self._input_queue.get_nowait()
                    self._input_queue.put_nowait(_ASYNC_DMTAC_CLOSE_SENTINEL)
                except queue.Empty:
                    pass
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        for q in (self._input_queue, self._event_queue, self._ready_queue):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass


def tactile_by_side_from_backends(
    active_sides: object,
    backend_for_side: Callable[[str], InspireTacHandBackend | None],
) -> dict[str, np.ndarray | None]:
    tactile_by_side: dict[str, np.ndarray | None] = {"left": None, "right": None}
    for side in tuple(str(side) for side in active_sides):
        backend = backend_for_side(side)
        if backend is None:
            continue
        tactile = backend.get_state().get("tactile")
        tactile_by_side[side] = None if tactile is None else np.asarray(tactile, dtype=float).reshape(-1)
    return tactile_by_side


def dmtac_force_by_side_from_backends(
    active_sides: object,
    backend_for_side: Callable[[str], Any | None],
) -> dict[str, dict[str, np.ndarray] | None]:
    tactile_by_side: dict[str, dict[str, np.ndarray] | None] = {"left": None, "right": None}
    for side in tuple(str(side) for side in active_sides):
        backend = backend_for_side(side)
        if backend is None:
            continue
        get_state = getattr(backend, "get_state")
        code = getattr(get_state, "__code__", None)
        arg_count = int(getattr(code, "co_argcount", 0) or 0) + int(
            getattr(code, "co_kwonlyargcount", 0) or 0
        )
        if code is not None and "include_tactile_vector" in code.co_varnames[:arg_count]:
            state = get_state(include_tactile_vector=False)
        else:
            state = get_state()
        tactile_dict = state.get("tactile_dict")
        if not isinstance(tactile_dict, dict):
            tactile_by_side[side] = None
            continue
        tactile_by_side[side] = {
            str(key): np.asarray(value, dtype=np.float32)
            for key, value in tactile_dict.items()
            if value is not None
        }
    return tactile_by_side


def dmtac_visualization_snapshot_from_backends(
    active_sides: object,
    backend_for_side: Callable[[str], Any | None],
    *,
    now: float | None = None,
) -> tuple[
    dict[str, dict[str, np.ndarray] | None],
    dict[str, dict[str, Any]],
]:
    """Return cached force arrays plus explicit freshness for visualization."""

    current_time = time.time() if now is None else float(now)
    tactile_by_side: dict[str, dict[str, np.ndarray] | None] = {
        "left": None,
        "right": None,
    }
    health_by_side: dict[str, dict[str, Any]] = {}
    for side in tuple(str(side) for side in active_sides):
        backend = backend_for_side(side)
        if backend is None:
            health_by_side[side] = {"status": "unavailable", "message": "backend unavailable"}
            continue
        get_state = getattr(backend, "get_state")
        try:
            state = get_state(
                include_tactile_vector=False,
                copy_tactile_arrays=False,
            )
        except TypeError:
            state = get_state()
        tactile_dict = state.get("tactile_dict")
        field_by_key = state.get("tactile_field_by_key")
        field_by_key = field_by_key if isinstance(field_by_key, dict) else {}
        configured_fields = tuple(state.get("tactile_fields") or ())
        if isinstance(tactile_dict, dict):
            tactile_by_side[side] = {
                str(key): np.asarray(value, dtype=np.float32)
                for key, value in tactile_dict.items()
                if value is not None
                and (
                    str(field_by_key.get(str(key), "")) == "distributed_force"
                    or (
                        not field_by_key
                        and configured_fields == ("distributed_force",)
                    )
                )
            }
        timestamp = state.get("tactile_timestamp")
        try:
            age_sec = None if timestamp is None else max(0.0, current_time - float(timestamp))
        except (TypeError, ValueError):
            age_sec = None
        try:
            read_hz = float(state.get("tactile_read_hz") or 0.0)
        except (TypeError, ValueError):
            read_hz = 0.0
        max_age_sec = max(0.5, 2.0 / read_hz) if read_hz > 0.0 else 1.0
        worker_required = bool(state.get("tactile_process_read"))
        worker_alive = bool(state.get("tactile_worker_alive"))
        error = str(state.get("telemetry_error") or "")
        if error or (worker_required and not worker_alive):
            status = "disconnected"
            message = error or "worker disconnected"
        elif age_sec is None:
            status = "waiting"
            message = "waiting for tactile frame"
        elif age_sec > max_age_sec:
            status = "stale"
            message = f"stale age={age_sec:.2f}s"
        else:
            status = "ok"
            message = f"age={age_sec:.2f}s"
        health_by_side[side] = {
            "status": status,
            "message": message,
            "age_sec": age_sec,
            "max_age_sec": max_age_sec,
            "worker_alive": worker_alive,
            "error": error or None,
        }
    return tactile_by_side, health_by_side


def tactile_by_side_from_hand_states(
    active_sides: object,
    hand_states: dict[str, dict[str, Any] | None],
) -> dict[str, np.ndarray | None]:
    tactile_by_side: dict[str, np.ndarray | None] = {"left": None, "right": None}
    for side in tuple(str(side) for side in active_sides):
        state = hand_states.get(side) or {}
        tactile = state.get("tactile")
        tactile_by_side[side] = None if tactile is None else np.asarray(tactile, dtype=float).reshape(-1)
    return tactile_by_side

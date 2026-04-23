from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .backends.hands import InspireTacHandBackend


@dataclass(frozen=True)
class TactileVisualizerConfig:
    enabled_sides: tuple[str, ...] = ("left", "right")
    window_name: str = "inspire_hand_tactile_uv"
    image_size: tuple[int, int] = (640, 640)
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
        self._cv2.imshow(self.config.window_name, canvas)
        self._cv2.waitKey(1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cv2.destroyWindow(self.config.window_name)
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

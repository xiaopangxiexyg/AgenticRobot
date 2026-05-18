from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

@dataclass(frozen=True)
class TactileVisualizerConfig:
    enabled_sides: tuple[str, ...] = ("left", "right")
    window_name: str = "inspire_hand_tactile_uv"
    image_size: tuple[int, int] = (640, 640)
    resizable_window: bool = True
    render_scale: float = 2.0
    value_scale: float = 0.02
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
        if self.render_scale <= 0.0:
            raise ValueError("render_scale must be positive")
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
        if self.config.resizable_window:
            flag = self._cv2.WINDOW_NORMAL | getattr(self._cv2, "WINDOW_FREERATIO", 0)
        else:
            flag = self._cv2.WINDOW_AUTOSIZE
        self._cv2.namedWindow(self.config.window_name, flag)
        if self.config.resizable_window:
            height, width = canvas.shape[:2]
            self._cv2.resizeWindow(self.config.window_name, int(width), int(height))
        self._window_created = True

    def _scaled_canvas(self, canvas: np.ndarray) -> np.ndarray:
        scale = float(self.config.render_scale) if self.config.resizable_window else 1.0
        if scale <= 1.0:
            return canvas
        height, width = canvas.shape[:2]
        target_w = max(1, int(round(width * scale)))
        target_h = max(1, int(round(height * scale)))
        return self._cv2.resize(canvas, (target_w, target_h), interpolation=self._cv2.INTER_LINEAR)

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
        self._cv2.imshow(self.config.window_name, self._scaled_canvas(canvas))
        self._cv2.waitKey(1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cv2.destroyWindow(self.config.window_name)
        except Exception:
            pass


@dataclass(frozen=True)
class DexHandForceVisualizerConfig:
    enabled_sides: tuple[str, ...] = ("right",)
    window_name: str = "dexhand_fingertip_force"
    image_size: tuple[int, int] = (640, 640)
    resizable_window: bool = True
    render_scale: float = 1.5
    value_scale: float = 0.02
    panel_gap_px: int = 24
    panel_padding_px: int = 20
    circle_radius_min: int = 10
    circle_radius_max: int = 46
    arrow_length_min: float = 12.0
    arrow_length_max: float = 68.0
    line_color: tuple[int, int, int] = (112, 124, 136)
    outline_color: tuple[int, int, int] = (148, 162, 176)
    background_color: tuple[int, int, int] = (18, 21, 26)
    detail_background_color: tuple[int, int, int] = (26, 31, 38)

    def __post_init__(self) -> None:
        if self.value_scale <= 0.0:
            raise ValueError("value_scale must be positive")
        if self.render_scale <= 0.0:
            raise ValueError("render_scale must be positive")
        if self.circle_radius_min <= 0 or self.circle_radius_max <= 0:
            raise ValueError("circle radii must be positive")
        if self.circle_radius_min > self.circle_radius_max:
            raise ValueError("circle_radius_min cannot exceed circle_radius_max")
        if self.arrow_length_min < 0.0 or self.arrow_length_max < 0.0:
            raise ValueError("arrow lengths must be non-negative")
        if self.arrow_length_min > self.arrow_length_max:
            raise ValueError("arrow_length_min cannot exceed arrow_length_max")
        sides = tuple(str(side) for side in self.enabled_sides)
        invalid = [side for side in sides if side not in {"left", "right"}]
        if invalid:
            raise ValueError(f"unsupported DexHand visualizer side(s): {invalid}")


class DexHandForceVisualizer:
    _FINGER_ORDER: tuple[str, ...] = ("th", "ff", "mf", "rf", "lf")
    _FINGER_LABELS: dict[str, str] = {
        "th": "thumb",
        "ff": "index",
        "mf": "middle",
        "rf": "ring",
        "lf": "little",
    }
    _RIGHT_FINGER_LAYOUT: dict[str, tuple[float, float]] = {
        "th": (0.18, 0.56),
        "ff": (0.37, 0.33),
        "mf": (0.52, 0.26),
        "rf": (0.67, 0.33),
        "lf": (0.82, 0.47),
    }
    _TACTILE_DIM = 15

    def __init__(self, config: DexHandForceVisualizerConfig):
        self.config = config
        self._cv2 = TactileUVVisualizer._import_cv2()
        self._window_created = False
        self._closed = False
        self._color_lut = self._cv2.applyColorMap(
            np.arange(256, dtype=np.uint8).reshape(256, 1),
            getattr(self._cv2, "COLORMAP_TURBO", self._cv2.COLORMAP_JET),
        ).reshape(256, 3)

    def _put_text(
        self,
        image: np.ndarray,
        text: str,
        org: tuple[int, int],
        *,
        scale: float = 0.5,
        color: tuple[int, int, int] = (235, 239, 244),
    ) -> None:
        self._cv2.putText(
            image,
            text,
            org,
            self._cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            3,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            image,
            text,
            org,
            self._cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            self._cv2.LINE_AA,
        )

    def _panel_background(self) -> np.ndarray:
        width, height = self.config.image_size
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[...] = np.array(self.config.background_color, dtype=np.uint8)
        detail_top = int(round(height * 0.70))
        self._cv2.rectangle(
            panel,
            (0, detail_top),
            (width - 1, height - 1),
            tuple(int(v) for v in self.config.detail_background_color),
            thickness=-1,
        )
        return panel

    def _finger_center(self, side: str, finger: str, width: int, height: int) -> tuple[int, int]:
        nx, ny = self._RIGHT_FINGER_LAYOUT[finger]
        if side == "left":
            nx = 1.0 - nx
        x = int(round(nx * (width - 1)))
        y = int(round(ny * (height - 1)))
        return x, y

    def _strength(self, value: float) -> float:
        safe = max(float(value), 0.0)
        ref = max(float(self.config.value_scale), 1e-6)
        return float(np.clip(np.log1p(safe / ref) / np.log1p(10.0), 0.0, 1.0))

    def _color(self, strength: float) -> tuple[int, int, int]:
        idx = int(round(np.clip(float(strength), 0.0, 1.0) * 255.0))
        return tuple(int(v) for v in self._color_lut[idx].tolist())

    def _detail_rows(self, panel: np.ndarray, forces: np.ndarray) -> None:
        height, width = panel.shape[:2]
        top = int(round(height * 0.74))
        row_h = max(22, int(round((height - top - 14) / max(len(self._FINGER_ORDER), 1))))
        self._put_text(
            panel,
            "finger      Fx (N)      Fy (N)      Fz (N)      |Fxy|",
            (16, top - 10),
            scale=0.42,
            color=(191, 201, 214),
        )
        for row_index, finger in enumerate(self._FINGER_ORDER):
            fx, fy, fz = [float(v) for v in forces[row_index]]
            fxy = math.hypot(fx, fy)
            text = (
                f"{finger:<2} {self._FINGER_LABELS[finger]:<6} "
                f"{fx:>8.3f}   {fy:>8.3f}   {fz:>8.3f}   {fxy:>8.3f}"
            )
            y = top + row_index * row_h + 18
            self._put_text(panel, text, (16, y), scale=0.45)
        self._put_text(
            panel,
            f"scale={float(self.config.value_scale):.3g}N  mapping: circle=Fz, arrow=Fx/Fy",
            (16, height - 12),
            scale=0.42,
            color=(191, 201, 214),
        )

    def _draw_hand_outline(self, panel: np.ndarray, side: str) -> None:
        height, width = panel.shape[:2]
        hand_height = int(round(height * 0.72))

        def mirror_x(nx: float) -> float:
            return 1.0 - nx if side == "left" else nx

        points = [self._finger_center(side, finger, width, hand_height) for finger in self._FINGER_ORDER]
        palm_points = np.array(
            [
                points[0],
                (int(round(mirror_x(0.34) * (width - 1))), int(round(height * 0.64))),
                (int(round(mirror_x(0.52) * (width - 1))), int(round(height * 0.70))),
                (int(round(mirror_x(0.70) * (width - 1))), int(round(height * 0.64))),
                points[-1],
            ],
            dtype=np.int32,
        )
        self._cv2.polylines(
            panel,
            [palm_points.reshape(-1, 1, 2)],
            isClosed=False,
            color=tuple(int(v) for v in self.config.line_color),
            thickness=10,
            lineType=self._cv2.LINE_AA,
        )

    def _draw_panel(self, *, side: str, tactile_vec: np.ndarray | None) -> np.ndarray:
        panel = self._panel_background()
        height, width = panel.shape[:2]
        self._put_text(panel, f"{side} hand DexHand tactile", (16, 28), scale=0.7)
        self._draw_hand_outline(panel, side)

        if tactile_vec is None:
            self._put_text(panel, "no tactile", (16, 60), scale=0.52, color=(180, 180, 180))
            return panel

        values = np.asarray(tactile_vec, dtype=np.float32).reshape(-1)
        if values.shape[0] != self._TACTILE_DIM:
            self._put_text(panel, "invalid tactile shape", (16, 60), scale=0.52, color=(60, 90, 240))
            return panel

        safe = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).reshape(len(self._FINGER_ORDER), 3)
        for finger_index, finger in enumerate(self._FINGER_ORDER):
            cx, cy = self._finger_center(side, finger, width, int(round(height * 0.72)))
            fx, fy, fz = [float(v) for v in safe[finger_index]]
            fxy = math.hypot(fx, fy)
            fz_strength = self._strength(abs(fz))
            fxy_strength = self._strength(fxy)

            outline_radius = max(22, int(round(min(width, height) * 0.075)))
            fill_radius = int(
                round(
                    self.config.circle_radius_min
                    + math.sqrt(max(fz_strength, 0.0))
                    * (self.config.circle_radius_max - self.config.circle_radius_min)
                )
            )
            self._cv2.circle(
                panel,
                (cx, cy),
                outline_radius,
                tuple(int(v) for v in self.config.outline_color),
                thickness=2,
                lineType=self._cv2.LINE_AA,
            )
            self._cv2.circle(
                panel,
                (cx, cy),
                fill_radius,
                self._color(fz_strength),
                thickness=-1,
                lineType=self._cv2.LINE_AA,
            )
            if fxy > 1e-8:
                arrow_len = (
                    float(self.config.arrow_length_min)
                    + fxy_strength * (float(self.config.arrow_length_max) - float(self.config.arrow_length_min))
                )
                end_x = cx + (fx / fxy) * arrow_len
                end_y = cy - (fy / fxy) * arrow_len
                end_pt = (int(round(end_x)), int(round(end_y)))
                self._cv2.arrowedLine(
                    panel,
                    (cx, cy),
                    end_pt,
                    (0, 0, 0),
                    3,
                    self._cv2.LINE_AA,
                    tipLength=0.25,
                )
                self._cv2.arrowedLine(
                    panel,
                    (cx, cy),
                    end_pt,
                    (240, 244, 248),
                    1,
                    self._cv2.LINE_AA,
                    tipLength=0.25,
                )

            self._put_text(
                panel,
                self._FINGER_LABELS[finger],
                (cx - 34, max(18, cy - outline_radius - 14)),
                scale=0.45,
                color=(206, 216, 228),
            )
            self._put_text(
                panel,
                f"Fz {fz:.2f}  Fxy {fxy:.2f}",
                (cx - 54, min(int(round(height * 0.69)), cy + outline_radius + 28)),
                scale=0.42,
                color=(206, 216, 228),
            )

        self._detail_rows(panel, safe)
        return panel

    def _ensure_window(self, image: np.ndarray) -> None:
        if self._window_created:
            return
        if self.config.resizable_window:
            flag = self._cv2.WINDOW_NORMAL | getattr(self._cv2, "WINDOW_FREERATIO", 0)
        else:
            flag = self._cv2.WINDOW_AUTOSIZE
        self._cv2.namedWindow(self.config.window_name, flag)
        if self.config.resizable_window:
            height, width = image.shape[:2]
            self._cv2.resizeWindow(self.config.window_name, int(width), int(height))
        self._window_created = True

    def update(self, tactile_by_side: dict[str, np.ndarray | None]) -> None:
        if self._closed:
            return
        panels = [
            self._draw_panel(side=side, tactile_vec=tactile_by_side.get(side))
            for side in tuple(self.config.enabled_sides)
        ]
        if not panels:
            panels = [self._draw_panel(side="right", tactile_vec=None)]
        image = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]
        scale = float(self.config.render_scale) if self.config.resizable_window else 1.0
        if scale > 1.0:
            image = self._cv2.resize(
                image,
                (
                    max(1, int(round(image.shape[1] * scale))),
                    max(1, int(round(image.shape[0] * scale))),
                ),
                interpolation=self._cv2.INTER_LINEAR,
            )
        self._ensure_window(image)
        self._cv2.imshow(self.config.window_name, image)
        self._cv2.waitKey(1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cv2.destroyWindow(self.config.window_name)
        except Exception:
            pass


@dataclass(frozen=True)
class DMTacForceVisualizerConfig:
    enabled_sides: tuple[str, ...] = ("right",)
    window_name: str = "dmtac_gripper_distributed_force"
    render_scale: float = 1.0
    force_channel: str = "xy"
    value_scale: float = 0.02
    arrow_stride: int = 24
    arrow_scale: float = 32.0
    arrow_pool: int = 16
    min_arrow_norm: float = 0.03
    show_stats: bool = True
    colorbar_width: int = 36
    resizable_window: bool = True

    def __post_init__(self) -> None:
        if self.render_scale <= 0.0:
            raise ValueError("render_scale must be positive")
        if self.value_scale <= 0.0:
            raise ValueError("value_scale must be positive")
        if self.arrow_stride <= 0:
            raise ValueError("arrow_stride must be positive")
        if self.arrow_scale < 0.0:
            raise ValueError("arrow_scale must be non-negative")
        if self.arrow_pool < 1:
            raise ValueError("arrow_pool must be positive")
        if self.min_arrow_norm < 0.0:
            raise ValueError("min_arrow_norm must be non-negative")
        if self.colorbar_width < 0:
            raise ValueError("colorbar_width must be non-negative")


class DMTacForceVisualizer:
    """OpenCV preview for DM-Tac getDistributeForce() gripper tactile frames."""

    def __init__(self, config: DMTacForceVisualizerConfig):
        self.config = config
        self._cv2 = TactileUVVisualizer._import_cv2()
        self._prev_force_by_label: dict[str, np.ndarray] = {}
        self._window_created = False
        self._closed = False

    def _blank_panel(self, title: str, message: str, color: tuple[int, int, int]) -> np.ndarray:
        panel = np.zeros((288, 384 + int(self.config.colorbar_width), 3), dtype=np.uint8)
        self._put_text(panel, f"{title} {message}".strip(), (8, 24), scale=0.55, color=color)
        return panel

    def _put_text(
        self,
        image: np.ndarray,
        text: str,
        org: tuple[int, int],
        *,
        scale: float = 0.48,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self._cv2.putText(
            image,
            text,
            org,
            self._cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            3,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            image,
            text,
            org,
            self._cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            self._cv2.LINE_AA,
        )

    def _pooled_xy_vector(self, xy: np.ndarray, *, x: int, y: int) -> tuple[float, float]:
        pool = max(1, int(self.config.arrow_pool))
        if pool <= 1:
            return float(xy[y, x, 0]), float(xy[y, x, 1])
        half = max(1, pool // 2)
        y1 = max(0, y - half)
        y2 = min(xy.shape[0], y + half + 1)
        x1 = max(0, x - half)
        x2 = min(xy.shape[1], x + half + 1)
        vec = np.nanmean(xy[y1:y2, x1:x2], axis=(0, 1))
        return float(vec[0]), float(vec[1])

    def _draw_xy_arrows(self, panel: np.ndarray, xy: np.ndarray, mag: np.ndarray) -> None:
        ref = max(float(self.config.value_scale), 1e-8)
        stride = max(1, int(self.config.arrow_stride))
        max_arrow_px = max(0.0, float(self.config.arrow_scale))
        min_arrow_norm = max(0.0, float(self.config.min_arrow_norm))
        if max_arrow_px <= 0.0:
            return
        color_lut = self._cv2.applyColorMap(
            np.arange(256, dtype=np.uint8).reshape(256, 1),
            getattr(self._cv2, "COLORMAP_TURBO", self._cv2.COLORMAP_JET),
        ).reshape(256, 3)
        height, width = mag.shape
        for y in range(stride // 2, height, stride):
            for x in range(stride // 2, width, stride):
                vx, vy = self._pooled_xy_vector(xy, x=x, y=y)
                length = float(np.hypot(vx, vy))
                if length <= 1e-8:
                    continue
                norm = min(length / ref, 1.0)
                if norm < min_arrow_norm:
                    continue
                arrow_len = max(1.0, norm * max_arrow_px)
                dx = vx / length * arrow_len
                dy = vy / length * arrow_len
                start = (int(round(x)), int(round(y)))
                end = (int(round(x + dx)), int(round(y + dy)))
                color = tuple(int(c) for c in color_lut[int(round(norm * 255.0))].tolist())
                self._cv2.arrowedLine(panel, start, end, (0, 0, 0), 3, self._cv2.LINE_AA, tipLength=0.35)
                self._cv2.arrowedLine(panel, start, end, color, 1, self._cv2.LINE_AA, tipLength=0.35)

    def _colorbar(self, height: int) -> np.ndarray | None:
        width = int(self.config.colorbar_width)
        if width <= 0:
            return None
        ramp = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
        bar = np.repeat(ramp, width, axis=1)
        out = self._cv2.applyColorMap(
            bar,
            getattr(self._cv2, "COLORMAP_TURBO", self._cv2.COLORMAP_JET),
        )
        self._put_text(out, f"{float(self.config.value_scale):.3g}", (3, 16), scale=0.36)
        self._put_text(out, "0", (3, height - 7), scale=0.36)
        return out

    def _force_to_panel(self, force: np.ndarray | None, title: str) -> np.ndarray:
        if force is None:
            return self._blank_panel(title, "no force", (210, 210, 210))
        arr = np.asarray(force, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 2:
            return self._blank_panel(title, "invalid", (60, 90, 240))
        channel = str(self.config.force_channel).lower()
        safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        xy = safe[..., :2]
        if channel == "z" and arr.shape[2] >= 3:
            mag = np.abs(safe[..., 2])
        else:
            mag = np.linalg.norm(xy, axis=2)
        scaled = np.clip(mag / float(self.config.value_scale), 0.0, 1.0)
        panel = self._cv2.applyColorMap(
            np.rint(scaled * 255.0).astype(np.uint8),
            getattr(self._cv2, "COLORMAP_TURBO", self._cv2.COLORMAP_JET),
        )
        if channel != "z":
            self._draw_xy_arrows(panel, xy, mag)
        self._put_text(panel, title, (8, 24), scale=0.55)
        if self.config.show_stats:
            prev = self._prev_force_by_label.get(title)
            dxy_max = 0.0
            if prev is not None and prev.shape == safe.shape:
                delta_xy = np.linalg.norm((safe - prev)[..., :2], axis=2)
                dxy_max = float(np.nanmax(delta_xy)) if delta_xy.size else 0.0
            stats = (
                f"{channel.upper()} mean={float(np.nanmean(mag)):.3g} "
                f"max={float(np.nanmax(mag)):.3g} dXYmax={dxy_max:.3g}"
            )
            self._put_text(panel, stats, (8, panel.shape[0] - 10), scale=0.45)
            self._prev_force_by_label[title] = safe.copy()
        colorbar = self._colorbar(panel.shape[0])
        if colorbar is not None:
            panel = np.concatenate([panel, colorbar], axis=1)
        return panel

    def _ensure_window(self, image: np.ndarray) -> None:
        if self._window_created:
            return
        if self.config.resizable_window:
            flag = self._cv2.WINDOW_NORMAL | getattr(self._cv2, "WINDOW_FREERATIO", 0)
        else:
            flag = self._cv2.WINDOW_AUTOSIZE
        self._cv2.namedWindow(self.config.window_name, flag)
        if self.config.resizable_window:
            height, width = image.shape[:2]
            self._cv2.resizeWindow(self.config.window_name, int(width), int(height))
        self._window_created = True

    def update(self, force_by_label: dict[str, np.ndarray | None]) -> None:
        if self._closed:
            return
        panels = [
            self._force_to_panel(force, label)
            for label, force in sorted(force_by_label.items())
        ]
        if not panels:
            panels = [self._force_to_panel(None, "no tactile")]
        height = max(panel.shape[0] for panel in panels)
        resized = []
        for panel in panels:
            if panel.shape[0] == height:
                resized.append(panel)
                continue
            width = int(round(panel.shape[1] * height / panel.shape[0]))
            resized.append(self._cv2.resize(panel, (width, height), interpolation=self._cv2.INTER_LINEAR))
        image = np.concatenate(resized, axis=1)
        scale = float(self.config.render_scale) if self.config.resizable_window else 1.0
        if scale > 1.0:
            image = self._cv2.resize(
                image,
                (
                    max(1, int(round(image.shape[1] * scale))),
                    max(1, int(round(image.shape[0] * scale))),
                ),
                interpolation=self._cv2.INTER_LINEAR,
            )
        self._ensure_window(image)
        self._cv2.imshow(self.config.window_name, image)
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
    backend_for_side: Callable[[str], Any | None],
) -> dict[str, np.ndarray | None]:
    tactile_by_side: dict[str, np.ndarray | None] = {"left": None, "right": None}
    for side in tuple(str(side) for side in active_sides):
        backend = backend_for_side(side)
        if backend is None:
            continue
        tactile = backend.get_state().get("tactile")
        tactile_by_side[side] = None if tactile is None else np.asarray(tactile, dtype=float).reshape(-1)
    return tactile_by_side


def dmtac_force_by_label_from_backends(
    active_sides: object,
    backend_for_side: Callable[[str], Any | None],
) -> dict[str, np.ndarray | None]:
    force_by_label: dict[str, np.ndarray | None] = {}
    for side in tuple(str(side) for side in active_sides):
        backend = backend_for_side(side)
        if backend is None:
            continue
        tactile_dict = backend.get_state().get("tactile_dict")
        if not isinstance(tactile_dict, dict) or not tactile_dict:
            force_by_label[f"{side}:no_sensor"] = None
            continue
        for dev_id, force in tactile_dict.items():
            force_by_label[f"{side}:{dev_id}"] = np.asarray(force, dtype=np.float32)
    return force_by_label

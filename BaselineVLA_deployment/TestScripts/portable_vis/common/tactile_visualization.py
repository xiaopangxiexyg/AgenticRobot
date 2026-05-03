from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Optional

import cv2
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

from .tactile_layout import (
    BEND_ID_LEFT,
    BEND_ID_RIGHT,
    HandConfig,
    HandType,
    INDEX_P_LEFT,
    INDEX_P_RIGHT,
    LEFT_ORDER,
    MIDDLE_P_LEFT,
    MIDDLE_P_RIGHT,
    PALM_P_LEFT,
    PALM_P_RIGHT,
    PINKY_P_LEFT,
    PINKY_P_RIGHT,
    RING_P_LEFT,
    RING_P_RIGHT,
    THUMB_P_LEFT,
    THUMB_P_RIGHT,
)


RIGHT_ORDER = THUMB_P_RIGHT + INDEX_P_RIGHT + MIDDLE_P_RIGHT + RING_P_RIGHT + PINKY_P_RIGHT + BEND_ID_RIGHT + PALM_P_RIGHT
_ORDER_BY_HAND = {
    "left": LEFT_ORDER,
    "right": RIGHT_ORDER,
}

_INFERNO_POSITIONS = np.asarray([0.0, 0.13, 0.25, 0.38, 0.50, 0.63, 0.75, 0.88, 1.0], dtype=np.float32)
_INFERNO_RGB = np.asarray(
    [
        [0, 0, 4],
        [31, 12, 72],
        [85, 15, 109],
        [136, 34, 106],
        [186, 54, 85],
        [227, 89, 51],
        [249, 140, 10],
        [249, 201, 50],
        [252, 255, 164],
    ],
    dtype=np.float32,
)
_INFERNO_BGR = _INFERNO_RGB[:, ::-1].copy()


@lru_cache(maxsize=2)
def _get_sparse_hand_config(hand_side: str) -> HandConfig:
    hand_name = str(hand_side).strip().lower()
    if hand_name not in _ORDER_BY_HAND:
        raise ValueError(f"Unsupported hand side: {hand_side}")
    config = HandConfig(HandType.LEFT if hand_name == "left" else HandType.RIGHT)
    config.id2pos = {
        int(sensor_id): index
        for index, sensor_id in enumerate(_ORDER_BY_HAND[hand_name])
        if int(sensor_id) > 0
    }
    return config


def tactile_vector_to_layout_matrices(values: np.ndarray, hand_side: str) -> tuple[list[np.ndarray], np.ndarray]:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    config = _get_sparse_hand_config(hand_side)
    expected_dim = len(_ORDER_BY_HAND[str(hand_side).strip().lower()])
    if vector.shape[0] != expected_dim:
        raise ValueError(f"Expected tactile vector length {expected_dim}, got {vector.shape[0]}")
    finger_mats = [config.extract_finger_val_mat(finger_index, vector) for finger_index in range(5)]
    palm_mat = config.extract_palm_val_mat(vector)
    return finger_mats, palm_mat


def compute_tactile_visualization_limits(
    values: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if mask.size == array.shape[0]:
            array = array[mask]
    if array.size == 0:
        return 0.0, 1.0

    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(finite_values, lower_percentile))
    vmax = float(np.percentile(finite_values, upper_percentile))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin + 1e-6:
        vmax = vmin + 1.0
    return vmin, vmax


def _colorize_matrix(matrix: np.ndarray, vmin: float, vmax: float, colormap: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    nan_mask = ~np.isfinite(values)
    denom = max(float(vmax - vmin), 1e-6)
    normalized = np.clip((np.nan_to_num(values, nan=vmin) - vmin) / denom, 0.0, 1.0)
    idx_hi = np.searchsorted(_INFERNO_POSITIONS, normalized, side="right")
    idx_hi = np.clip(idx_hi, 1, len(_INFERNO_POSITIONS) - 1)
    idx_lo = idx_hi - 1
    pos_lo = _INFERNO_POSITIONS[idx_lo]
    pos_hi = _INFERNO_POSITIONS[idx_hi]
    span = np.maximum(pos_hi - pos_lo, 1e-6)
    weight_hi = ((normalized - pos_lo) / span)[..., None]
    weight_lo = 1.0 - weight_hi
    color = _INFERNO_BGR[idx_lo] * weight_lo + _INFERNO_BGR[idx_hi] * weight_hi
    color = np.clip(np.round(color), 0.0, 255.0).astype(np.uint8)
    color[nan_mask] = np.array([40, 40, 40], dtype=np.uint8)
    return color


def _resize_nearest_numpy(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source = np.ascontiguousarray(np.asarray(image))
    src_h, src_w = source.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError("Cannot resize an empty image.")
    dst_w = max(int(width), 1)
    dst_h = max(int(height), 1)
    x_idx = np.minimum((np.arange(dst_w, dtype=np.int64) * src_w) // dst_w, src_w - 1)
    y_idx = np.minimum((np.arange(dst_h, dtype=np.int64) * src_h) // dst_h, src_h - 1)
    return np.ascontiguousarray(source[y_idx][:, x_idx])


def _safe_int_point(xy: Any) -> Optional[tuple[int, int]]:
    point = np.asarray(xy, dtype=np.float64).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _draw_hline(canvas: np.ndarray, x0: int, x1: int, y: int, color: tuple[int, int, int], thickness: int = 1) -> None:
    height, width = canvas.shape[:2]
    if height <= 0 or width <= 0:
        return
    x0 = max(0, min(int(x0), width))
    x1 = max(0, min(int(x1), width))
    if x0 > x1:
        x0, x1 = x1, x0
    half = max(int(thickness), 1) // 2
    y_start = max(0, int(y) - half)
    y_end = min(height, int(y) + half + 1)
    if y_start >= y_end or x0 >= x1:
        return
    canvas[y_start:y_end, x0:x1] = np.asarray(color, dtype=canvas.dtype)


def _draw_vline(canvas: np.ndarray, x: int, y0: int, y1: int, color: tuple[int, int, int], thickness: int = 1) -> None:
    height, width = canvas.shape[:2]
    if height <= 0 or width <= 0:
        return
    y0 = max(0, min(int(y0), height))
    y1 = max(0, min(int(y1), height))
    if y0 > y1:
        y0, y1 = y1, y0
    half = max(int(thickness), 1) // 2
    x_start = max(0, int(x) - half)
    x_end = min(width, int(x) + half + 1)
    if x_start >= x_end or y0 >= y1:
        return
    canvas[y0:y1, x_start:x_end] = np.asarray(color, dtype=canvas.dtype)


def _blend_mask_bgr(image: np.ndarray, mask: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    if mask_h <= 0 or mask_w <= 0:
        return

    x0 = max(int(x), 0)
    y0 = max(int(y), 0)
    x1 = min(int(x) + mask_w, width)
    y1 = min(int(y) + mask_h, height)
    if x0 >= x1 or y0 >= y1:
        return

    mask_x0 = x0 - int(x)
    mask_y0 = y0 - int(y)
    mask_crop = np.asarray(mask[mask_y0 : mask_y0 + (y1 - y0), mask_x0 : mask_x0 + (x1 - x0)], dtype=np.float32)
    if mask_crop.size <= 0:
        return
    alpha = (mask_crop / 255.0)[..., None]
    if not np.any(alpha > 0.0):
        return

    region = image[y0:y1, x0:x1].astype(np.float32, copy=False)
    color_vec = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    blended = region * (1.0 - alpha) + color_vec * alpha
    image[y0:y1, x0:x1] = np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8)


@lru_cache(maxsize=1)
def _get_default_pil_font() -> Any:
    if not _PIL_AVAILABLE:
        return None
    return ImageFont.load_default()


def _draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    point = _safe_int_point(org)
    if point is None:
        return
    safe_color = tuple(int(np.clip(channel, 0, 255)) for channel in color)

    if _PIL_AVAILABLE:
        font = _get_default_pil_font()
        if font is not None:
            probe = Image.new("L", (1, 1), 0)
            probe_draw = ImageDraw.Draw(probe)
            stroke_width = max(0, int(thickness) - 1)
            try:
                bbox = probe_draw.textbbox((0, 0), str(text), font=font, stroke_width=stroke_width)
            except TypeError:
                bbox = probe_draw.textbbox((0, 0), str(text), font=font)
                stroke_width = 0
            text_w = max(int(bbox[2] - bbox[0]), 1)
            text_h = max(int(bbox[3] - bbox[1]), 1)
            mask_img = Image.new("L", (text_w, text_h), 0)
            draw = ImageDraw.Draw(mask_img)
            try:
                draw.text((-bbox[0], -bbox[1]), str(text), fill=255, font=font, stroke_width=stroke_width, stroke_fill=255)
            except TypeError:
                draw.text((-bbox[0], -bbox[1]), str(text), fill=255, font=font)
            scale_factor = max(1, int(round(float(scale) * 2.0)))
            if scale_factor > 1:
                resample = getattr(getattr(Image, "Resampling", Image), "NEAREST")
                mask_img = mask_img.resize((text_w * scale_factor, text_h * scale_factor), resample=resample)
            mask = np.asarray(mask_img, dtype=np.uint8)
            _blend_mask_bgr(image, mask, point[0], point[1], safe_color)
            return

    safe_image = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    try:
        cv2.putText(
            safe_image,
            str(text),
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            float(scale),
            safe_color,
            int(thickness),
            lineType=cv2.LINE_AA,
        )
    except Exception:
        return
    if safe_image is not image:
        np.copyto(image, safe_image)


def _draw_matrix_block(
    canvas: np.ndarray,
    matrix: np.ndarray,
    title: str,
    x: int,
    y: int,
    cell_size: int,
    vmin: float,
    vmax: float,
    colormap: int,
) -> None:
    rows, cols = matrix.shape
    block = _colorize_matrix(matrix, vmin=vmin, vmax=vmax, colormap=colormap)
    block = _resize_nearest_numpy(block, cols * cell_size, rows * cell_size)
    canvas[y:y + block.shape[0], x:x + block.shape[1]] = block
    for row in range(rows + 1):
        line_y = y + row * cell_size
        _draw_hline(canvas, x, x + cols * cell_size, line_y, (16, 16, 16), 1)
    for col in range(cols + 1):
        line_x = x + col * cell_size
        _draw_vline(canvas, line_x, y, y + rows * cell_size, (16, 16, 16), 1)
    _draw_text(canvas, title, (x, max(18, y - 8)), (235, 235, 235), 0.55, 1)


def render_single_hand_tactile_panel(
    hand_side: str,
    tactile_payload: Optional[Mapping[str, Any]],
    width: int = 320,
    height: int = 320,
    vmin: float = 0.0,
    vmax: float = 1.0,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    panel = np.full((int(height), int(width), 3), 22, dtype=np.uint8)
    hand_name = str(hand_side).strip().lower()
    title = hand_name.upper()
    _draw_text(panel, title, (16, 28), (250, 250, 250), 0.8, 2)

    if tactile_payload is None:
        _draw_text(panel, "NO TACTILE", (16, 62), (160, 160, 160), 0.6, 1)
        return np.ascontiguousarray(panel)

    values = np.asarray(tactile_payload.get("values"), dtype=np.float32).reshape(-1)
    valid = bool(tactile_payload.get("valid", False))
    interpolated = bool(tactile_payload.get("interpolated", False))
    delta_ms = float(tactile_payload.get("time_delta_ms", float("nan")))
    source_frame_index = int(tactile_payload.get("source_frame_index", -1))
    status = "valid"
    status_color = (70, 210, 120)
    if interpolated:
        status = "interp"
        status_color = (0, 215, 255)
    elif not valid:
        status = "invalid"
        status_color = (60, 60, 255)
    _draw_text(panel, status, (16, 58), status_color, 0.6, 2)
    _draw_text(panel, f"src={source_frame_index} dt={delta_ms:.1f}ms", (16, 82), (210, 210, 210), 0.48, 1)

    finger_mats, palm_mat = tactile_vector_to_layout_matrices(values, hand_name)
    finger_titles = ["thumb", "index", "middle", "ring", "pinky"]
    margin = 14
    available_width = max(width - margin * 2, 10)
    finger_gap = 6
    finger_block_width = max((available_width - finger_gap * 4) // 5, 15)
    cell_size_finger = max(min(finger_block_width // 3, 16), 8)
    x = margin
    y = 112
    for finger_title, finger_mat in zip(finger_titles, finger_mats):
        _draw_matrix_block(panel, finger_mat, finger_title, x, y, cell_size_finger, vmin, vmax, colormap)
        x += finger_mat.shape[1] * cell_size_finger + finger_gap

    palm_top = y + finger_mats[0].shape[0] * cell_size_finger + 34
    cell_size_palm = max(min((available_width // palm_mat.shape[1]), 16), 8)
    _draw_matrix_block(panel, palm_mat, "palm", margin, palm_top, cell_size_palm, vmin, vmax, colormap)
    return np.ascontiguousarray(panel)


def render_combined_tactile_panel(
    tactile_by_hand: Mapping[str, Mapping[str, Any]],
    limits_by_hand: Optional[Mapping[str, tuple[float, float]]] = None,
    total_width: int = 640,
    target_height: int = 720,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    panel_width = max(int(total_width // 2), 160)
    panel_height = max(int(target_height), 160)
    panels = []
    for hand_side in ("left", "right"):
        limits = (0.0, 1.0) if limits_by_hand is None else limits_by_hand.get(hand_side, (0.0, 1.0))
        panels.append(
            render_single_hand_tactile_panel(
                hand_side=hand_side,
                tactile_payload=tactile_by_hand.get(hand_side),
                width=panel_width,
                height=panel_height,
                vmin=float(limits[0]),
                vmax=float(limits[1]),
                colormap=colormap,
            )
        )
    return np.ascontiguousarray(np.concatenate(panels, axis=1))


__all__ = [
    "RIGHT_ORDER",
    "compute_tactile_visualization_limits",
    "render_combined_tactile_panel",
    "render_single_hand_tactile_panel",
    "tactile_vector_to_layout_matrices",
]

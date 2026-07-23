from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def open_resizable_window(
    cv2: Any,
    name: str,
    *,
    width: int = 1600,
    height: int = 900,
) -> None:
    flags = int(getattr(cv2, "WINDOW_NORMAL", 0))
    # Qt HighGUI's KEEPRATIO mode can leave a small fixed image widget centered
    # in a much larger white client area.  FREERATIO makes the widget occupy
    # the full client area; compose_previews preserves RGB aspect ratio itself
    # by rendering into a correctly sized letterboxed canvas.
    free_ratio = int(getattr(cv2, "WINDOW_FREERATIO", 0))
    flags |= free_ratio
    cv2.namedWindow(name, flags)
    set_property = getattr(cv2, "setWindowProperty", None)
    aspect_property = getattr(cv2, "WND_PROP_ASPECT_RATIO", None)
    if set_property is not None and aspect_property is not None:
        try:
            set_property(name, int(aspect_property), free_ratio)
        except Exception:
            pass
    requested_width = max(320, int(width))
    requested_height = max(240, int(height))
    # Some HighGUI/Qt backends ignore resizeWindow before the first imshow.
    # Realize the native window with a full-size placeholder first so the
    # requested geometry is applied to the actual image client area.
    cv2.imshow(
        name,
        np.zeros((requested_height, requested_width, 3), dtype=np.uint8),
    )
    cv2.resizeWindow(name, requested_width, requested_height)
    cv2.waitKey(1)


def compose_previews(
    cv2: Any,
    previews: Mapping[str, np.ndarray],
    *,
    max_width: int = 1600,
    max_height: int = 900,
) -> np.ndarray:
    if not previews:
        raise ValueError("at least one preview is required")
    images = [np.asarray(image) for image in previews.values()]
    if any(image.ndim != 3 or image.shape[2] != 3 for image in images):
        raise ValueError("previews must be HxWx3 images")
    common_height = min(image.shape[0] for image in images)
    normalized = [
        _resize_to_height(cv2, image, common_height)
        if image.shape[0] != common_height
        else image
        for image in images
    ]
    canvas = normalized[0] if len(normalized) == 1 else np.concatenate(normalized, axis=1)
    scale = min(
        float(max_width) / float(canvas.shape[1]),
        float(max_height) / float(canvas.shape[0]),
    )
    size = (
        max(1, int(round(canvas.shape[1] * scale))),
        max(1, int(round(canvas.shape[0] * scale))),
    )
    if abs(scale - 1.0) < 0.001:
        resized = np.ascontiguousarray(canvas)
    else:
        interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
        resized = np.ascontiguousarray(
            cv2.resize(canvas, size, interpolation=interpolation)
        )
    target_width = max(1, int(max_width))
    target_height = max(1, int(max_height))
    if resized.shape[1] == target_width and resized.shape[0] == target_height:
        return resized
    output = np.zeros((target_height, target_width, 3), dtype=resized.dtype)
    x = max(0, (target_width - resized.shape[1]) // 2)
    y = max(0, (target_height - resized.shape[0]) // 2)
    output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return np.ascontiguousarray(output)


def compose_dashboard(
    cv2: Any,
    previews: Mapping[str, np.ndarray],
    status_lines: Mapping[str, list[str]],
    ready_by_side: Mapping[str, bool],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if not previews:
        raise ValueError("at least one preview is required")
    target_width = max(320, int(width))
    target_height = max(240, int(height))
    output = np.full((target_height, target_width, 3), 18, dtype=np.uint8)
    sides = list(previews)
    column_width = target_width // len(sides)
    status_height = min(
        max(190, int(round(target_height * 0.28))),
        max(190, target_height - 180),
    )
    image_height = max(1, target_height - status_height)
    for index, side in enumerate(sides):
        x0 = index * column_width
        x1 = target_width if index + 1 == len(sides) else (index + 1) * column_width
        panel_width = x1 - x0
        _fit_image_into_region(
            cv2,
            output,
            np.asarray(previews[side]),
            x=x0,
            y=0,
            width=panel_width,
            height=image_height,
        )
        cv2.rectangle(
            output,
            (x0, image_height),
            (x1 - 1, target_height - 1),
            (28, 28, 28),
            -1,
        )
        if index > 0:
            cv2.line(output, (x0, 0), (x0, target_height), (85, 85, 85), 2)
        _draw_dashboard_text(
            cv2,
            output,
            side=str(side),
            lines=list(status_lines.get(side, [])),
            ready=bool(ready_by_side.get(side, False)),
            x=x0,
            y=image_height,
            width=panel_width,
            height=status_height,
        )
    return np.ascontiguousarray(output)


def current_window_render_size(
    cv2: Any,
    name: str,
    *,
    fallback_width: int,
    fallback_height: int,
) -> tuple[int, int]:
    fallback = (max(1, int(fallback_width)), max(1, int(fallback_height)))
    get_rect = getattr(cv2, "getWindowImageRect", None)
    if get_rect is None:
        return fallback
    try:
        _, _, width, height = get_rect(name)
    except Exception:
        return fallback
    width = int(width)
    height = int(height)
    if width < 64 or height < 64:
        return fallback
    return width, height


def draw_status_lines(
    cv2: Any,
    image: np.ndarray,
    lines: list[str],
    *,
    ready: bool,
) -> None:
    if not lines:
        return
    line_height = 23
    top = max(0, image.shape[0] - line_height * len(lines) - 10)
    cv2.rectangle(image, (0, top), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    color = (80, 240, 80) if ready else (80, 210, 255)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            str(line)[:150],
            (12, top + 21 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )


def _fit_image_into_region(
    cv2: Any,
    output: np.ndarray,
    image: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("preview must be an HxWx3 image")
    scale = min(float(width) / image.shape[1], float(height) / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    left = x + max(0, (width - resized_width) // 2)
    top = y + max(0, (height - resized_height) // 2)
    output[top : top + resized_height, left : left + resized_width] = resized


def _draw_dashboard_text(
    cv2: Any,
    output: np.ndarray,
    *,
    side: str,
    lines: list[str],
    ready: bool,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    density = min(float(width) / 800.0, float(output.shape[0]) / 900.0)
    body_scale = max(0.62, min(0.90, 0.76 * density))
    title_scale = max(0.78, min(1.10, body_scale * 1.22))
    title_color = (90, 245, 90) if ready else (80, 215, 255)
    cv2.putText(
        output,
        f"{side.upper()}  {'READY' if ready else 'COLLECTING'}",
        (x + 18, y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        title_color,
        2,
        cv2.LINE_AA,
    )
    line_height = max(28, int(round(34 * body_scale / 0.76)))
    cursor_y = y + 70
    max_y = y + height - 12
    for line in lines:
        for wrapped in _wrap_text(cv2, str(line), width - 36, body_scale, 2):
            if cursor_y > max_y:
                return
            cv2.putText(
                output,
                wrapped,
                (x + 18, cursor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                body_scale,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            cursor_y += line_height


def _wrap_text(
    cv2: Any,
    text: str,
    max_width: int,
    font_scale: float,
    thickness: int,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        size, _ = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        if int(size[0]) <= int(max_width):
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _resize_to_height(cv2: Any, image: np.ndarray, height: int) -> np.ndarray:
    scale = float(height) / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, int(height)), interpolation=cv2.INTER_AREA)


__all__ = [
    "compose_dashboard",
    "compose_previews",
    "current_window_render_size",
    "draw_status_lines",
    "open_resizable_window",
]

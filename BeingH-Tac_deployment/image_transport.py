from __future__ import annotations

from typing import Any

import numpy as np


def resize_short_side_center_crop(image: Any, size: int) -> np.ndarray:
    target = int(size)
    arr = np.asarray(image)
    if target <= 0:
        return arr.astype(np.uint8, copy=False)
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected HxWxC image, got shape={arr.shape}.")
    if arr.shape[0] == target and arr.shape[1] == target:
        return arr.astype(np.uint8, copy=False)

    try:
        import cv2
    except Exception as exc:
        raise ImportError("cv2 is required for client-side image resizing.") from exc

    height, width = int(arr.shape[0]), int(arr.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape={arr.shape}.")
    scale = float(target) / float(min(height, width))
    resized_h = max(target, int(round(height * scale)))
    resized_w = max(target, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(arr, (resized_w, resized_h), interpolation=interpolation)
    y0 = max(0, (resized_h - target) // 2)
    x0 = max(0, (resized_w - target) // 2)
    cropped = resized[y0 : y0 + target, x0 : x0 + target]
    return np.ascontiguousarray(cropped.astype(np.uint8, copy=False))


def maybe_resize_payload_image(payload: dict[str, Any], size: int) -> None:
    if int(size) <= 0:
        return
    for key in ("image", "front_image", "image.front", "observation.images.front", "video.front"):
        if key in payload and payload[key] is not None:
            payload[key] = resize_short_side_center_crop(payload[key], int(size))
            return
    images = payload.get("images")
    if isinstance(images, dict) and images.get("front") is not None:
        images["front"] = resize_short_side_center_crop(images["front"], int(size))

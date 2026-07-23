"""RealSense color transport conversion helpers.

RealSense UVC color sensors commonly expose YUYV as their native transport
format.  The project-facing camera APIs still use ordinary three-channel BGR
or RGB images, so the conversion belongs at the hardware boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SUPPORTED_COLOR_FORMATS = frozenset(
    {"bgr8", "rgb8", "yuyv", "uyvy", "y8", "rgba8", "bgra8"}
)


def normalize_color_format(color_format: str) -> str:
    normalized = str(color_format).strip().lower()
    if normalized not in SUPPORTED_COLOR_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_COLOR_FORMATS))
        raise ValueError(
            f"unsupported RealSense color_format {color_format!r}; "
            f"expected one of: {supported}"
        )
    return normalized


def color_format_value(rs: Any, color_format: str) -> Any:
    normalized = normalize_color_format(color_format)
    value = getattr(rs.format, normalized, None)
    if value is None:
        raise RuntimeError(f"pyrealsense2 does not expose color format {normalized!r}")
    return value


def color_image_to_bgr(color_image: np.ndarray, color_format: str) -> np.ndarray:
    """Convert one RealSense color payload to a contiguous BGR image.

    pyrealsense2 2.58 exposes a YUYV frame as packed ``uint16[h, w]`` on this
    host, while other bindings expose ``uint8[h, w, 2]``.  Both layouts are
    accepted without changing the byte order.
    """

    image = np.asanyarray(color_image)
    normalized = normalize_color_format(color_format)
    if normalized == "bgr8":
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"bgr8 color frame has unexpected shape {image.shape}")
        return np.ascontiguousarray(image).copy()

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - all live camera paths use OpenCV
        raise RuntimeError(
            "OpenCV is required to convert non-BGR RealSense color frames."
        ) from exc

    contiguous = np.ascontiguousarray(image)
    if normalized in {"yuyv", "uyvy"}:
        if contiguous.dtype != np.uint8:
            if contiguous.ndim != 2 or contiguous.dtype.itemsize != 2:
                raise RuntimeError(
                    f"{normalized} color frame must be uint8[h,w,2] or packed "
                    f"uint16[h,w]; got dtype={contiguous.dtype}, shape={contiguous.shape}"
                )
            contiguous = contiguous.view(np.uint8).reshape(
                contiguous.shape[0], contiguous.shape[1], 2
            )
        if contiguous.ndim == 2:
            if contiguous.shape[1] % 2 != 0:
                raise RuntimeError(
                    f"{normalized} color frame has unexpected shape {contiguous.shape}"
                )
            contiguous = contiguous.reshape(contiguous.shape[0], contiguous.shape[1] // 2, 2)
        if contiguous.ndim != 3 or contiguous.shape[2] != 2:
            raise RuntimeError(
                f"{normalized} color frame has unexpected shape {contiguous.shape}"
            )
        conversion = (
            cv2.COLOR_YUV2BGR_YUY2
            if normalized == "yuyv"
            else cv2.COLOR_YUV2BGR_UYVY
        )
        return cv2.cvtColor(contiguous, conversion)
    if normalized == "rgb8":
        return cv2.cvtColor(contiguous, cv2.COLOR_RGB2BGR)
    if normalized == "rgba8":
        return cv2.cvtColor(contiguous, cv2.COLOR_RGBA2BGR)
    if normalized == "bgra8":
        return cv2.cvtColor(contiguous, cv2.COLOR_BGRA2BGR)
    if normalized == "y8":
        return cv2.cvtColor(contiguous, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported RealSense color format {color_format!r}")

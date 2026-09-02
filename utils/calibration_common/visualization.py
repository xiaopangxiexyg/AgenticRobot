"""Pure point-cloud projection shared by calibration visualizers."""

from __future__ import annotations

from typing import Any

import numpy as np


def depth_to_numpy_cloud(
    color_bgr: np.ndarray,
    depth_z16: np.ndarray,
    *,
    intrinsics_matrix: np.ndarray,
    depth_scale_m: float,
    depth_trunc_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = np.asarray(depth_z16, dtype=np.float64) * float(depth_scale_m)
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= float(depth_trunc_m))
    valid_indices = np.flatnonzero(valid.reshape(-1))
    if valid_indices.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    max_points = max(1, int(max_points))
    if valid_indices.size > max_points:
        pick = np.linspace(0, valid_indices.size - 1, max_points).astype(np.int64)
        valid_indices = valid_indices[pick]

    height, width = depth_m.shape
    rows = valid_indices // width
    cols = valid_indices % width
    z = depth_m.reshape(-1)[valid_indices]
    fx = float(intrinsics_matrix[0, 0])
    fy = float(intrinsics_matrix[1, 1])
    cx = float(intrinsics_matrix[0, 2])
    cy = float(intrinsics_matrix[1, 2])
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    points = np.column_stack((x, y, z))
    color_rgb = np.asarray(color_bgr[:, :, ::-1], dtype=np.float64).reshape(height * width, 3)
    colors = color_rgb[valid_indices] / 255.0
    return points, colors


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def draw_projection_views(
    cv2: Any,
    points_base: np.ndarray,
    colors_rgb: np.ndarray,
    *,
    T_camera_to_base: np.ndarray,
    panel_size: int,
) -> np.ndarray:
    bounds = projection_bounds(points_base)
    panels = [
        draw_projection_panel(
            cv2,
            points_base,
            colors_rgb,
            dims=(0, 1),
            labels=("X", "Y"),
            title="base XY",
            bounds=bounds,
            panel_size=panel_size,
            T_camera_to_base=T_camera_to_base,
        ),
        draw_projection_panel(
            cv2,
            points_base,
            colors_rgb,
            dims=(0, 2),
            labels=("X", "Z"),
            title="base XZ",
            bounds=bounds,
            panel_size=panel_size,
            T_camera_to_base=T_camera_to_base,
        ),
        draw_projection_panel(
            cv2,
            points_base,
            colors_rgb,
            dims=(1, 2),
            labels=("Y", "Z"),
            title="base YZ",
            bounds=bounds,
            panel_size=panel_size,
            T_camera_to_base=T_camera_to_base,
        ),
    ]
    return np.concatenate(panels, axis=1)


def projection_bounds(points: np.ndarray) -> tuple[np.ndarray, float]:
    mins = np.percentile(points, 2.0, axis=0)
    maxs = np.percentile(points, 98.0, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(0.1, float(np.max(maxs - mins)) * 0.6)
    return center, radius


def draw_projection_panel(
    cv2: Any,
    points: np.ndarray,
    colors_rgb: np.ndarray,
    *,
    dims: tuple[int, int],
    labels: tuple[str, str],
    title: str,
    bounds: tuple[np.ndarray, float],
    panel_size: int,
    T_camera_to_base: np.ndarray,
) -> np.ndarray:
    margin = 36
    image = np.full((panel_size, panel_size, 3), 24, dtype=np.uint8)
    center, radius = bounds
    xy = project_points_to_panel(
        points,
        dims=dims,
        center=center,
        radius=radius,
        panel_size=panel_size,
        margin=margin,
    )
    colors_bgr = np.clip(colors_rgb[:, ::-1] * 255.0, 0, 255).astype(np.uint8)
    valid = (
        (xy[:, 0] >= margin)
        & (xy[:, 0] < panel_size - margin)
        & (xy[:, 1] >= margin)
        & (xy[:, 1] < panel_size - margin)
    )
    if np.any(valid):
        image[xy[valid, 1], xy[valid, 0]] = colors_bgr[valid]

    cv2.rectangle(image, (margin, margin), (panel_size - margin, panel_size - margin), (85, 85, 85), 1)
    draw_frame_projection(cv2, image, np.eye(4), dims=dims, center=center, radius=radius, panel_size=panel_size, margin=margin, name="base")
    draw_frame_projection(cv2, image, T_camera_to_base, dims=dims, center=center, radius=radius, panel_size=panel_size, margin=margin, name="cam")
    cv2.putText(image, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"{labels[0]} / {labels[1]}  {len(points)} pts",
        (12, panel_size - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return image


def project_points_to_panel(
    points: np.ndarray,
    *,
    dims: tuple[int, int],
    center: np.ndarray,
    radius: float,
    panel_size: int,
    margin: int,
) -> np.ndarray:
    scale = (panel_size - 2 * margin) / (2.0 * float(radius))
    px = margin + (points[:, dims[0]] - center[dims[0]] + radius) * scale
    py = panel_size - margin - (points[:, dims[1]] - center[dims[1]] + radius) * scale
    return np.column_stack((np.rint(px), np.rint(py))).astype(np.int32)


def draw_frame_projection(
    cv2: Any,
    image: np.ndarray,
    transform: np.ndarray,
    *,
    dims: tuple[int, int],
    center: np.ndarray,
    radius: float,
    panel_size: int,
    margin: int,
    name: str,
) -> None:
    origin = transform[:3, 3]
    axes = transform[:3, :3] * 0.12
    colors = ((0, 0, 255), (0, 220, 0), (255, 80, 40))
    labels = ("x", "y", "z")
    origin_px = project_points_to_panel(
        origin.reshape(1, 3),
        dims=dims,
        center=center,
        radius=radius,
        panel_size=panel_size,
        margin=margin,
    )[0]
    for index, color in enumerate(colors):
        end = origin + axes[:, index]
        end_px = project_points_to_panel(
            end.reshape(1, 3),
            dims=dims,
            center=center,
            radius=radius,
            panel_size=panel_size,
            margin=margin,
        )[0]
        cv2.line(image, tuple(origin_px), tuple(end_px), color, 2, cv2.LINE_AA)
        cv2.putText(image, f"{name}.{labels[index]}", tuple(end_px), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


__all__ = [
    "depth_to_numpy_cloud",
    "draw_frame_projection",
    "draw_projection_panel",
    "draw_projection_views",
    "project_points_to_panel",
    "projection_bounds",
    "transform_points",
]

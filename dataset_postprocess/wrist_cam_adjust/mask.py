from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataset_postprocess.real_undistort.intrinsics import CameraIntrinsics, require_cv2

from .geometry import TableGeometry, invert_pose


@dataclass(frozen=True)
class RayGrid:
    width: int
    height: int
    rays_camera: np.ndarray
    full_width: int
    full_height: int
    stride: int = 1


def build_ray_grid(intrinsics: CameraIntrinsics, *, stride: int = 1) -> RayGrid:
    """Return one optical-frame ray per distorted image pixel."""

    cv2 = require_cv2()
    full_width = int(intrinsics.width)
    full_height = int(intrinsics.height)
    stride = max(1, int(stride))
    width = int(np.ceil(float(full_width) / float(stride)))
    height = int(np.ceil(float(full_height) / float(stride)))
    if stride == 1:
        x_coords = np.arange(width, dtype=np.float32)
        y_coords = np.arange(height, dtype=np.float32)
    else:
        offset = (float(stride) - 1.0) * 0.5
        x_coords = np.clip(np.arange(width, dtype=np.float32) * float(stride) + offset, 0.0, full_width - 1.0)
        y_coords = np.clip(np.arange(height, dtype=np.float32) * float(stride) + offset, 0.0, full_height - 1.0)
    xs, ys = np.meshgrid(x_coords, y_coords)
    pixels = np.stack((xs, ys), axis=-1).reshape(-1, 1, 2)
    coeffs = intrinsics.dist_coeffs()
    if coeffs.size and np.any(np.abs(coeffs) > 1e-12):
        normalized = cv2.undistortPoints(
            pixels,
            intrinsics.camera_matrix(),
            coeffs,
            P=None,
        ).reshape(height, width, 2)
        rays = np.dstack((normalized[..., 0], normalized[..., 1], np.ones((height, width), dtype=np.float64)))
    else:
        rays = np.dstack(
            (
                (xs.astype(np.float64) - float(intrinsics.ppx)) / float(intrinsics.fx),
                (ys.astype(np.float64) - float(intrinsics.ppy)) / float(intrinsics.fy),
                np.ones((height, width), dtype=np.float64),
            )
        )
    return RayGrid(
        width=width,
        height=height,
        rays_camera=rays.astype(np.float64, copy=False),
        full_width=full_width,
        full_height=full_height,
        stride=stride,
    )


def table_keep_mask(
    *,
    table: TableGeometry,
    T_world_camera: np.ndarray,
    ray_grid: RayGrid,
    table_margin_m: float = 0.0,
    table_height_offset_m: float = 0.0,
) -> np.ndarray:
    """Project a finite URDF table top into the camera image."""

    T_world_camera = np.asarray(T_world_camera, dtype=float).reshape(4, 4)
    T_visual_camera = table.T_visual_world @ T_world_camera
    camera_origin_visual = T_visual_camera[:3, 3]
    rays_visual = ray_grid.rays_camera @ T_visual_camera[:3, :3].T

    table_z = float(table.top_z_visual) + float(table_height_offset_m)
    denominator = rays_visual[..., 2]
    valid = np.abs(denominator) > 1e-9
    distance = np.zeros((ray_grid.height, ray_grid.width), dtype=np.float64)
    np.divide(
        table_z - float(camera_origin_visual[2]),
        denominator,
        out=distance,
        where=valid,
    )
    valid &= distance > 0.0

    points_x = float(camera_origin_visual[0]) + distance * rays_visual[..., 0]
    points_y = float(camera_origin_visual[1]) + distance * rays_visual[..., 1]
    half_x = float(table.size[0]) * 0.5 + float(table_margin_m)
    half_y = float(table.size[1]) * 0.5 + float(table_margin_m)
    on_table = (
        valid
        & (np.abs(points_x) <= half_x)
        & (np.abs(points_y) <= half_y)
    )
    return np.where(on_table, 255, 0).astype(np.uint8)


def table_perimeter_points(
    *,
    table: TableGeometry,
    table_margin_m: float = 0.0,
    table_height_offset_m: float = 0.0,
    edge_samples: int = 16,
) -> np.ndarray:
    """Return sampled finite-table perimeter points in world coordinates."""

    samples = max(2, int(edge_samples))
    half_x = max(0.0, float(table.size[0]) * 0.5 + float(table_margin_m))
    half_y = max(0.0, float(table.size[1]) * 0.5 + float(table_margin_m))
    z = float(table.top_z_visual) + float(table_height_offset_m)

    xs = np.linspace(-half_x, half_x, samples, endpoint=False, dtype=np.float64)
    ys = np.linspace(-half_y, half_y, samples, endpoint=False, dtype=np.float64)
    points_visual = np.vstack(
        (
            np.column_stack((xs, np.full(samples, -half_y), np.full(samples, z))),
            np.column_stack((np.full(samples, half_x), ys, np.full(samples, z))),
            np.column_stack((-xs, np.full(samples, half_y), np.full(samples, z))),
            np.column_stack((np.full(samples, -half_x), -ys, np.full(samples, z))),
        )
    )
    return points_visual @ table.T_world_visual[:3, :3].T + table.T_world_visual[:3, 3].reshape(1, 3)


def _clip_polygon_min_z(points_camera: np.ndarray, min_depth_m: float) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3:
        return np.zeros((0, 3), dtype=np.float64)
    min_z = float(min_depth_m)
    clipped: list[np.ndarray] = []
    previous = points[-1]
    previous_inside = bool(previous[2] >= min_z)
    for current in points:
        current_inside = bool(current[2] >= min_z)
        if current_inside != previous_inside:
            dz = float(current[2] - previous[2])
            if abs(dz) > 1e-12:
                ratio = (min_z - float(previous[2])) / dz
                clipped.append(previous + ratio * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    if len(clipped) < 3:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64).reshape(-1, 3)


def table_keep_mask_polygon(
    *,
    table_points_world: np.ndarray,
    T_world_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    min_depth_m: float = 0.03,
) -> np.ndarray | None:
    """Rasterize a sampled table top polygon into the camera image.

    This fast path is much cheaper than testing every image pixel against the
    table plane. It returns None when too few perimeter points are in front of
    the camera, allowing callers to fall back to ray casting for rare views.
    """

    cv2 = require_cv2()
    width = int(intrinsics.width)
    height = int(intrinsics.height)
    points_world = np.asarray(table_points_world, dtype=np.float64).reshape(-1, 3)
    if points_world.shape[0] < 3:
        return None

    T_camera_world = invert_pose(np.asarray(T_world_camera, dtype=float).reshape(4, 4))
    points_camera = points_world @ T_camera_world[:3, :3].T + T_camera_world[:3, 3].reshape(1, 3)
    if np.any(points_camera[:, 2] < float(min_depth_m)):
        return None
    clipped_camera = _clip_polygon_min_z(points_camera, float(min_depth_m))
    if clipped_camera.shape[0] < 3:
        return None

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    projected, _ = cv2.projectPoints(
        clipped_camera.reshape(-1, 1, 3),
        rvec,
        tvec,
        intrinsics.camera_matrix(),
        intrinsics.dist_coeffs(),
    )
    projected_points = projected.reshape(-1, 2)
    if not np.all(np.isfinite(projected_points)):
        return None
    clip = max(width, height) * 8
    projected_points[:, 0] = np.clip(projected_points[:, 0], -clip, width + clip)
    projected_points[:, 1] = np.clip(projected_points[:, 1], -clip, height + clip)
    points = np.rint(projected_points).astype(np.int32)
    if points.shape[0] < 3:
        return None
    points[:, 0] = np.clip(points[:, 0], -clip, width + clip)
    points[:, 1] = np.clip(points[:, 1], -clip, height + clip)
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    out = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(out, hull, 255)
    return out


def load_binary_mask(path: str | Path | None, *, width: int, height: int, label: str) -> np.ndarray:
    cv2 = require_cv2()
    if path is None:
        return np.zeros((int(height), int(width)), dtype=np.uint8)
    mask_path = Path(path).expanduser().resolve()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read {label} mask: {mask_path}")
    if mask.shape[:2] != (int(height), int(width)):
        mask = cv2.resize(mask, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    radius = int(pixels)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if radius <= 0:
        return binary
    cv2 = require_cv2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(binary, kernel)


def soften_mask(mask: np.ndarray, blur_px: int = 0, blur_sigma: float = 0.0) -> np.ndarray:
    alpha = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    radius = int(blur_px)
    if radius <= 0:
        return alpha
    cv2 = require_cv2()
    kernel_size = radius * 2 + 1
    return cv2.GaussianBlur(
        alpha,
        (kernel_size, kernel_size),
        float(blur_sigma),
        borderType=cv2.BORDER_REPLICATE,
    )


def combine_keep_mask(
    *,
    body_mask: np.ndarray,
    table_mask: np.ndarray,
    body_dilate_px: int = 0,
    table_dilate_px: int = 0,
    keep_dilate_px: int = 0,
) -> np.ndarray:
    body = dilate_mask(body_mask, body_dilate_px)
    table = dilate_mask(table_mask, table_dilate_px)
    keep = np.where((body > 0) | (table > 0), 255, 0).astype(np.uint8)
    return dilate_mask(keep, keep_dilate_px)


def background_mask_from_keep(keep_mask: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(keep_mask) > 0, 0, 255).astype(np.uint8)


def render_mask_output(
    frame: np.ndarray,
    *,
    keep_mask: np.ndarray,
    output_kind: str,
    background_color: tuple[int, int, int] = (0, 0, 0),
    blur_px: int = 0,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    cv2 = require_cv2()
    image = np.asarray(frame)
    kind = str(output_kind)
    if kind == "masked" and int(blur_px) <= 0:
        binary = np.where(np.asarray(keep_mask) > 0, 255, 0).astype(np.uint8)
        out = np.empty_like(image)
        out[:, :] = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
        cv2.copyTo(image, binary, out)
        return out

    keep_alpha_u8 = soften_mask(keep_mask, blur_px=blur_px, blur_sigma=blur_sigma)
    keep_alpha = keep_alpha_u8.astype(np.float32) / 255.0
    if kind == "masked":
        bg = np.empty_like(image)
        bg[:, :] = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
        alpha3 = keep_alpha[..., None]
        return np.clip(image.astype(np.float32) * alpha3 + bg.astype(np.float32) * (1.0 - alpha3), 0, 255).astype(np.uint8)
    if kind == "overlay":
        out = image.astype(np.float32, copy=True)
        red = np.zeros_like(out)
        red[:, :] = (0, 0, 255)
        background_alpha = (1.0 - keep_alpha)[..., None] * 0.65
        return np.clip(out * (1.0 - background_alpha) + red * background_alpha, 0, 255).astype(np.uint8)
    if kind == "background-mask":
        mask = np.clip(255 - keep_alpha_u8, 0, 255).astype(np.uint8)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if kind == "keep-mask":
        return cv2.cvtColor(keep_alpha_u8, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported output kind: {output_kind!r}")


def mask_stats(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask) > 0
    total = int(binary.size)
    count = int(np.count_nonzero(binary))
    return {
        "pixels": count,
        "total_pixels": total,
        "fraction": 0.0 if total <= 0 else float(count) / float(total),
    }

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


POINTCLOUD_FORMAT = "dexteleop_sim_pointcloud_npz_v1"


@dataclass(frozen=True)
class RenderCameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fovy_deg: float

    def camera_matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [float(self.fx), 0.0, float(self.cx)],
                [0.0, float(self.fy), float(self.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "fovy_deg": float(self.fovy_deg),
            "camera_matrix": self.camera_matrix().tolist(),
        }


@dataclass(frozen=True)
class PointCloudFrame:
    points_camera: np.ndarray
    points_world: np.ndarray
    uv: np.ndarray
    depth_m: np.ndarray
    valid_mask: np.ndarray
    intrinsics: RenderCameraIntrinsics
    T_world_camera: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.points_camera.shape[0])


def render_intrinsics_from_fovy(
    *,
    width: int,
    height: int,
    fovy_deg: float,
) -> RenderCameraIntrinsics:
    w = int(width)
    h = int(height)
    fovy = float(fovy_deg)
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be positive")
    if not math.isfinite(fovy) or fovy <= 0.0:
        raise ValueError(f"fovy_deg must be finite and positive, got {fovy_deg!r}")
    focal = float(h) / (2.0 * math.tan(math.radians(fovy) * 0.5))
    if not math.isfinite(focal) or focal <= 0.0:
        raise ValueError(f"invalid focal length from fovy_deg={fovy_deg!r}")
    return RenderCameraIntrinsics(
        width=w,
        height=h,
        fx=focal,
        fy=focal,
        cx=float(w) * 0.5,
        cy=float(h) * 0.5,
        fovy_deg=fovy,
    )


def pointcloud_from_depth(
    depth_m: np.ndarray,
    *,
    fovy_deg: float,
    T_world_camera: np.ndarray,
    depth_min_m: float = 0.05,
    depth_max_m: float = 1.0,
    pixel_stride: int = 1,
) -> PointCloudFrame:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be 2D, got shape={depth.shape!r}")
    height, width = int(depth.shape[0]), int(depth.shape[1])
    intrinsics = render_intrinsics_from_fovy(width=width, height=height, fovy_deg=float(fovy_deg))
    stride = max(1, int(pixel_stride))
    sampled_depth = depth[::stride, ::stride]
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= np.float32(depth_min_m))
        & (sampled_depth <= np.float32(depth_max_m))
    )
    if not np.any(valid):
        empty_points = np.zeros((0, 3), dtype=np.float32)
        return PointCloudFrame(
            points_camera=empty_points,
            points_world=empty_points.copy(),
            uv=np.zeros((0, 2), dtype=np.int32),
            depth_m=np.zeros((0,), dtype=np.float32),
            valid_mask=valid.astype(np.uint8, copy=False),
            intrinsics=intrinsics,
            T_world_camera=np.asarray(T_world_camera, dtype=np.float64).reshape(4, 4),
        )

    z = sampled_depth[valid].astype(np.float32, copy=False)
    u = xs[valid].astype(np.float32, copy=False)
    v = ys[valid].astype(np.float32, copy=False)
    x = (u - np.float32(intrinsics.cx)) * z / np.float32(intrinsics.fx)
    y = (v - np.float32(intrinsics.cy)) * z / np.float32(intrinsics.fy)
    points_camera = np.column_stack((x, y, z)).astype(np.float32, copy=False)
    uv = np.column_stack((u.astype(np.int32), v.astype(np.int32))).astype(np.int32, copy=False)

    T = np.asarray(T_world_camera, dtype=np.float64).reshape(4, 4)
    points_world = (
        points_camera.astype(np.float64) @ T[:3, :3].T
        + T[:3, 3].reshape(1, 3)
    ).astype(np.float32, copy=False)
    return PointCloudFrame(
        points_camera=points_camera,
        points_world=points_world,
        uv=uv,
        depth_m=z.astype(np.float32, copy=False),
        valid_mask=valid.astype(np.uint8, copy=False),
        intrinsics=intrinsics,
        T_world_camera=T,
    )


def pointcloud_manifest_payload(
    *,
    camera_key: str,
    relative_dir: str,
    frame_count: int,
    records: list[Mapping[str, Any]],
    depth_min_m: float,
    depth_max_m: float,
    pixel_stride: int,
    coordinate_frames: tuple[str, ...] = ("camera_optical", "world"),
) -> dict[str, Any]:
    return {
        "format": POINTCLOUD_FORMAT,
        "camera_key": str(camera_key),
        "relative_dir": str(relative_dir),
        "encoding": "per_frame_npz",
        "frame_count": int(frame_count),
        "config": {
            "depth_min_m": float(depth_min_m),
            "depth_max_m": float(depth_max_m),
            "pixel_stride": int(pixel_stride),
            "coordinate_frames": list(coordinate_frames),
        },
        "semantics": {
            "points_camera": "float32 [N,3], camera optical frame: x right, y down, z forward",
            "points_world": "float32 [N,3], MuJoCo/URDF world frame",
            "uv": "int32 [N,2], image pixel coordinates [u,v] in the undistorted MuJoCo render image",
            "depth_m": "float32 [N], metric depth used for back-projection",
            "valid_mask": "uint8 [ceil(H/stride),ceil(W/stride)], sampled depth range mask",
            "source": "MuJoCo metric depth render, not decoded from depth MP4",
        },
        "frames": [dict(record) for record in records],
    }

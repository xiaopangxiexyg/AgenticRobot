from __future__ import annotations

import numpy as np

from sim.mujoco.pointcloud import pointcloud_from_depth, render_intrinsics_from_fovy


def test_render_intrinsics_from_fovy_uses_mujoco_center_projection() -> None:
    intrinsics = render_intrinsics_from_fovy(width=4, height=4, fovy_deg=90.0)

    assert intrinsics.width == 4
    assert intrinsics.height == 4
    np.testing.assert_allclose(intrinsics.fx, 2.0)
    np.testing.assert_allclose(intrinsics.fy, 2.0)
    np.testing.assert_allclose(intrinsics.cx, 2.0)
    np.testing.assert_allclose(intrinsics.cy, 2.0)


def test_pointcloud_from_depth_backprojects_camera_and_world_points() -> None:
    depth = np.ones((4, 4), dtype=np.float32)
    depth[0, 1] = 0.0
    depth[3, 3] = 2.0
    T_world_camera = np.eye(4, dtype=float)
    T_world_camera[:3, 3] = [0.5, -0.25, 1.0]

    pointcloud = pointcloud_from_depth(
        depth,
        fovy_deg=90.0,
        T_world_camera=T_world_camera,
        depth_min_m=0.5,
        depth_max_m=1.5,
    )

    assert pointcloud.point_count == 14
    center_index = np.where((pointcloud.uv[:, 0] == 2) & (pointcloud.uv[:, 1] == 2))[0]
    assert center_index.shape == (1,)
    np.testing.assert_allclose(pointcloud.points_camera[center_index[0]], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(pointcloud.points_world[center_index[0]], [0.5, -0.25, 2.0])
    assert not np.any((pointcloud.uv[:, 0] == 1) & (pointcloud.uv[:, 1] == 0))
    assert not np.any((pointcloud.uv[:, 0] == 3) & (pointcloud.uv[:, 1] == 3))


def test_pointcloud_from_depth_respects_pixel_stride() -> None:
    depth = np.ones((4, 4), dtype=np.float32)

    pointcloud = pointcloud_from_depth(
        depth,
        fovy_deg=90.0,
        T_world_camera=np.eye(4),
        depth_min_m=0.5,
        depth_max_m=1.5,
        pixel_stride=2,
    )

    assert pointcloud.point_count == 4
    assert set(map(tuple, pointcloud.uv.tolist())) == {(0, 0), (2, 0), (0, 2), (2, 2)}
    assert pointcloud.valid_mask.shape == (2, 2)

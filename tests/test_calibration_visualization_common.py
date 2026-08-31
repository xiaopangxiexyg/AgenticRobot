from __future__ import annotations

import numpy as np

import utils.calibration_common.visualization as common_visualization
import utils.hand_eye.visualize_handeye as hand_visualization
import utils.head_eye.visualize_handeye as head_visualization


def test_visualizer_namespaces_export_one_projection_implementation() -> None:
    for compatibility_name, shared_name in (
        ("_depth_to_numpy_cloud", "depth_to_numpy_cloud"),
        ("_draw_frame_projection", "draw_frame_projection"),
        ("_draw_projection_panel", "draw_projection_panel"),
        ("_draw_projection_views", "draw_projection_views"),
        ("_project_points_to_panel", "project_points_to_panel"),
        ("_projection_bounds", "projection_bounds"),
        ("_transform_points", "transform_points"),
    ):
        shared = getattr(common_visualization, shared_name)
        assert getattr(hand_visualization, compatibility_name) is shared
        assert getattr(head_visualization, compatibility_name) is shared


def test_depth_projection_preserves_points_and_rgb_values() -> None:
    color_bgr = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    depth_z16 = np.array([[1000, 0], [2000, 4000]], dtype=np.uint16)
    intrinsics = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    points, colors = common_visualization.depth_to_numpy_cloud(
        color_bgr,
        depth_z16,
        intrinsics_matrix=intrinsics,
        depth_scale_m=0.001,
        depth_trunc_m=3.0,
        max_points=10,
    )

    np.testing.assert_allclose(points, [[0.0, 0.0, 1.0], [0.0, 2.0, 2.0]])
    np.testing.assert_allclose(colors * 255.0, [[30, 20, 10], [90, 80, 70]])


def test_projection_views_remain_three_fixed_size_panels() -> None:
    points = np.array(
        [[-0.1, 0.0, 0.2], [0.0, 0.1, 0.3], [0.1, -0.1, 0.4]],
        dtype=np.float64,
    )
    colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    image = common_visualization.draw_projection_views(
        _FakeCV2,
        points,
        colors,
        T_camera_to_base=np.eye(4),
        panel_size=120,
    )

    assert image.shape == (120, 360, 3)
    assert image.dtype == np.uint8


class _FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    @staticmethod
    def rectangle(image, start, end, color, thickness) -> None:
        del start, end, color, thickness
        assert image.ndim == 3

    @staticmethod
    def line(image, start, end, color, thickness, line_type) -> None:
        del start, end, color, thickness, line_type
        assert image.ndim == 3

    @staticmethod
    def putText(image, text, origin, font, scale, color, thickness, line_type) -> None:
        del text, origin, font, scale, color, thickness, line_type
        assert image.ndim == 3

from __future__ import annotations

import numpy as np

from deployments.perception.dmtac_grasp_overlay import (
    DmtacGraspOverlayCandidate,
    camera_from_reference_poses,
    project_dmtac_grasp,
    render_dmtac_grasp_overlays,
)

INTRINSICS = {
    "width": 100,
    "height": 80,
    "fx": 100.0,
    "fy": 100.0,
    "cx": 50.0,
    "cy": 40.0,
}


def _candidate(*, x: float = 0.0, z: float = 1.0) -> DmtacGraspOverlayCandidate:
    reference_from_tcp = np.eye(4)
    reference_from_tcp[:3, 3] = [x, 0.0, z]
    return DmtacGraspOverlayCandidate.from_pose(
        ordinal=0,
        score=0.9,
        gripper_width_m=0.04,
        reference_from_tcp=reference_from_tcp,
        selected=True,
    )


def test_reference_camera_transform_preserves_pose_chain() -> None:
    reference_from_camera = np.eye(4)
    reference_from_camera[:3, 3] = [0.2, -0.1, 0.3]
    reference_from_tcp = np.eye(4)
    reference_from_tcp[:3, 3] = [-0.3, 0.4, 0.8]

    camera_from_tcp = camera_from_reference_poses(
        reference_from_tcp=reference_from_tcp,
        reference_from_camera=reference_from_camera,
    )

    np.testing.assert_allclose(
        reference_from_camera @ camera_from_tcp,
        reference_from_tcp,
        atol=1e-12,
    )


def test_projection_classifies_full_partial_outside_and_behind() -> None:
    visible = project_dmtac_grasp(
        _candidate(),
        reference_from_camera=np.eye(4),
        intrinsics=INTRINSICS,
    )
    partial = project_dmtac_grasp(
        _candidate(x=0.48),
        reference_from_camera=np.eye(4),
        intrinsics=INTRINSICS,
    )
    outside = project_dmtac_grasp(
        _candidate(x=5.0),
        reference_from_camera=np.eye(4),
        intrinsics=INTRINSICS,
    )
    behind = project_dmtac_grasp(
        _candidate(z=-1.0),
        reference_from_camera=np.eye(4),
        intrinsics=INTRINSICS,
    )

    assert visible.visibility == "visible"
    assert visible.visible_segment_count == visible.total_segment_count == 39
    assert visible.tcp_pixel_uv == (50, 40)
    assert partial.visibility == "partially_visible"
    assert 0 < partial.visible_segment_count < partial.total_segment_count
    assert outside.visibility == "out_of_frame"
    assert outside.front_segment_count == outside.total_segment_count
    assert outside.visible_segment_count == 0
    assert behind.visibility == "behind_camera"
    assert behind.front_segment_count == behind.visible_segment_count == 0


def test_near_plane_and_image_clipping_keep_pixels_in_bounds() -> None:
    candidate = _candidate(z=0.01)
    projection = project_dmtac_grasp(
        candidate,
        reference_from_camera=np.eye(4),
        intrinsics=INTRINSICS,
    )

    assert projection.visibility == "partially_visible"
    assert projection.clipped_segment_count > 0
    assert projection.visible_segment_count > 0
    for segment in projection.segments:
        for u, v in (segment.start_uv, segment.end_uv):
            assert 0 <= u < INTRINSICS["width"]
            assert 0 <= v < INTRINSICS["height"]


def test_overlay_draws_both_finger_boxes_palm_axes_and_visibility_legend() -> None:
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    image, projections = render_dmtac_grasp_overlays(
        rgb,
        candidates=(_candidate(), _candidate(x=5.0)),
        reference_from_camera=np.eye(4),
        intrinsics=INTRINSICS,
        target_pixel_uv=(50, 40),
        target_bbox_xyxy=(40, 30, 60, 50),
    )

    rendered = np.asarray(image)
    assert rendered.shape == rgb.shape
    assert int(rendered.sum()) > 0
    assert {segment.kind for segment in projections[0].segments} >= {
        "left_finger",
        "right_finger",
        "palm",
        "axis_x",
        "axis_y",
        "axis_z",
    }
    assert projections[1].visibility == "out_of_frame"
    assert projections[1].visible_segment_count == 0
    assert projections[1].to_dict()["T_camera_from_tcp"][3] == 5.0

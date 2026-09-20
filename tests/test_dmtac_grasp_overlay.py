from __future__ import annotations

import numpy as np
import pytest

from deployments.perception.dmtac_grasp_overlay import (
    DMTAC_M_FINGER_LENGTH_M,
    DMTAC_M_GRASPGENX_FINGER_LENGTH_M,
    DMTAC_M_GRASPGENX_FINGER_THICKNESS_M,
    DMTAC_M_GRASPGENX_MAX_OPENING_M,
    DMTAC_M_GRASPGENX_PAD_HEIGHT_M,
    DMTAC_M_GRASPGENX_PALM_Z_M,
    DMTAC_M_PALM_FROM_TOOL_TCP_Z_M,
    DmtacGraspOverlayCandidate,
    _wireframe_segments,
    camera_from_reference_poses,
    project_dmtac_grasp,
    render_dmtac_grasp_overlays,
    reviewed_graspgenx_dmtac_solid_geometry,
    reviewed_4b2_cad_solid_geometry,
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


def test_wireframe_is_expressed_about_actual_tool_tcp() -> None:
    segments = _wireframe_segments(0.04)
    finger_points = [
        point
        for kind, first, second in segments
        if kind in {"left_finger", "right_finger"}
        for point in (first, second)
    ]
    z_values = np.asarray([point[2] for point in finger_points])

    assert z_values.min() == DMTAC_M_PALM_FROM_TOOL_TCP_Z_M
    assert z_values.max() == (DMTAC_M_PALM_FROM_TOOL_TCP_Z_M + DMTAC_M_FINGER_LENGTH_M)
    assert z_values.min() < 0.0 < z_values.max()


def test_cad_geometry_moves_body_not_controlled_tcp_axes() -> None:
    candidate = _candidate()
    geometry = reviewed_4b2_cad_solid_geometry()
    rendered = project_dmtac_grasp(candidate, reference_from_camera=np.eye(4),
                                  intrinsics=INTRINSICS, solid_geometry=geometry)
    legacy = project_dmtac_grasp(candidate, reference_from_camera=np.eye(4),
                                intrinsics=INTRINSICS)
    assert rendered.tcp_pixel_uv == legacy.tcp_pixel_uv == (50, 40)
    assert [s for s in rendered.segments if s.kind.startswith("axis_")] == [
        s for s in legacy.segments if s.kind.startswith("axis_")]
    assert [s for s in rendered.segments if not s.kind.startswith("axis_")] != [
        s for s in legacy.segments if not s.kind.startswith("axis_")]
    # The flange mating offset lifts the 97.001 mm internal marker to
    # 105.201 mm; the real controller still controls its calibrated 140 mm TCP.
    assert geometry.solid_pose(np.eye(4))[2, 3] + geometry.palm_z_m == pytest.approx(-.034799)


def test_graspgenx_dmtac_profile_is_separate_from_legacy_4b2_profile() -> None:
    legacy = reviewed_4b2_cad_solid_geometry()
    graspgenx = reviewed_graspgenx_dmtac_solid_geometry()

    assert legacy.maximum_opening_m == pytest.approx(0.070)
    assert legacy.finger_length_m == pytest.approx(0.050)
    assert legacy.finger_thickness_m == pytest.approx(0.010)
    assert legacy.pad_height_m == pytest.approx(0.0255)
    assert graspgenx.maximum_opening_m == pytest.approx(DMTAC_M_GRASPGENX_MAX_OPENING_M)
    assert graspgenx.palm_z_m == pytest.approx(DMTAC_M_GRASPGENX_PALM_Z_M)
    assert graspgenx.finger_length_m == pytest.approx(DMTAC_M_GRASPGENX_FINGER_LENGTH_M)
    assert graspgenx.finger_thickness_m == pytest.approx(DMTAC_M_GRASPGENX_FINGER_THICKNESS_M)
    assert graspgenx.pad_height_m == pytest.approx(DMTAC_M_GRASPGENX_PAD_HEIGHT_M)
    assert graspgenx != legacy
    assert graspgenx.solid_pose(np.eye(4)).shape == (4, 4)


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

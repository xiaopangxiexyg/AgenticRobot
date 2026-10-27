"""Measured depth support geometry is a proposal, never a motion permit."""

import numpy as np
import pytest
from dataclasses import replace
from scipy.spatial.transform import Rotation

from agent_gateway.profiles.back_projection import (
    LiveAlignedDepthCapture, project_depth_mask_support_surface,
    search_depth_mask_support_surface, check_observed_support_footprint,
    bind_known_box_support_from_depth,
)
from robot_motion.object_transport import CalibratedBoxSupportObservation


def capture(*, depth=None, tilt_deg=0.):
    if depth is None:
        depth = np.ones((20, 20), dtype=np.float32)
    transform = np.eye(4)
    transform[:3, :3] = (Rotation.from_euler("y", tilt_deg, degrees=True).as_matrix()
                         @ np.diag((1., -1., -1.)))
    transform[:3, 3] = (0., 0., 1.)
    return LiveAlignedDepthCapture(
        source_id="registered-main", role="main", captured_at=10.,
        station_revision=3, capture_group_id="same-cut:1",
        depth_content_hash="0" * 64,
        intrinsics=dict(width=20, height=20, fx=100., fy=100., cx=9.5, cy=9.5),
        reference_from_camera=transform, depth_m=depth)


def mask():
    result = np.zeros((20, 20), dtype=bool)
    result[1:-1, 1:-1] = True
    return result


def test_native_metric_depth_returns_inner_flat_top_and_conservative_height():
    cut = capture()
    result = project_depth_mask_support_surface(
        cut, mask(), mask_clipped=False, maximum_depth_m=2.)
    assert result.upper_top_z_m == pytest.approx(.002)
    assert result.normal_reference_xyz == pytest.approx((0., 0., 1.))
    assert result.observed_bounds_min_xy_m[0] >= -.085
    assert result.observed_bounds_max_xy_m[0] <= .085
    assert result.top_point_count == 18 * 18
    assert result.maximum_plane_residual_m < 1e-10
    assert cut.depth_m.flags.writeable is False


def test_lower_side_pixels_do_not_become_support_top():
    depth = np.ones((20, 20), dtype=np.float32)
    depth[1:3, 1:-1] = 1.1
    result = project_depth_mask_support_surface(
        capture(depth=depth), mask(), mask_clipped=False, maximum_depth_m=2.)
    assert result.upper_top_z_m == pytest.approx(.002)
    assert result.top_point_count < 18 * 18


@pytest.mark.parametrize("change,match", [
    (dict(mask_clipped=True), "Clipped"),
    (dict(maximum_depth_m=.5), "insufficient"),
    (dict(maximum_tilt_deg=2.), "exceed"),
    (dict(maximum_plane_residual_m=.01), "exceed"),
])
def test_support_projection_rejects_unqualified_source_or_profile(change, match):
    options = dict(mask_clipped=False, maximum_depth_m=2.)
    options.update(change)
    with pytest.raises(ValueError, match=match):
        project_depth_mask_support_surface(capture(), mask(), **options)


def test_tilted_support_is_not_labeled_as_horizontal():
    with pytest.raises(ValueError, match="near-horizontal"):
        project_depth_mask_support_surface(
            capture(tilt_deg=2.), mask(), mask_clipped=False, maximum_depth_m=2.)


def test_search_finds_observed_local_plane_without_promoting_to_complete_region():
    depth = np.ones((20, 20), dtype=np.float32)
    depth[1:4, 1:4] = .90  # A raised object contaminates the wide mask.
    result = search_depth_mask_support_surface(
        capture(depth=depth), mask(), mask_clipped=False,
        maximum_depth_m=2., patch_size_px=16, stride_px=2)
    assert result.accepted_patch_count > 0
    assert result.inspected_patch_count >= result.accepted_patch_count
    assert result.candidate.top_point_count >= 32
    assert result.candidate.observed_bounds_max_xy_m[0] > result.candidate.observed_bounds_min_xy_m[0]
    assert not hasattr(result.candidate, "geometry_complete")


@pytest.mark.parametrize("options, match", [
    ({"mask_clipped": True}, "Clipped"),
    ({"maximum_patches": 1}, "budget"),
    ({"patch_size_px": True}, "positive integer"),
    ({"maximum_depth_m": True}, "maximum_depth_m"),
])
def test_search_rejects_unqualified_or_unbounded_input(options, match):
    values = dict(mask_clipped=False, maximum_depth_m=2., patch_size_px=16,
                  stride_px=2)
    values.update(options)
    with pytest.raises(ValueError, match=match):
        search_depth_mask_support_surface(capture(), mask(), **values)


def test_dense_footprint_coverage_accepts_measured_flat_interior():
    cut = capture()
    selected = mask()
    plane = project_depth_mask_support_surface(
        cut, selected, mask_clipped=False, maximum_depth_m=2.)
    result = check_observed_support_footprint(
        cut, selected, plane,
        region_min_xy_m=(-.04, -.04), region_max_xy_m=(.04, .04))
    assert result.sample_count > 1000
    assert result.maximum_measured_plane_residual_m < 1e-9


def test_dense_footprint_coverage_rejects_hole_and_raised_occluder():
    cut = capture()
    plane = project_depth_mask_support_surface(
        cut, mask(), mask_clipped=False, maximum_depth_m=2.)
    gap = mask()
    gap[10, 10] = False
    with pytest.raises(ValueError, match="mask gap"):
        check_observed_support_footprint(
            cut, gap, plane, region_min_xy_m=(-.04, -.04),
            region_max_xy_m=(.04, .04))
    depth = np.ones((20, 20), dtype=np.float32)
    depth[10, 10] = .99
    with pytest.raises(ValueError, match="nonplanar"):
        check_observed_support_footprint(
            capture(depth=depth), mask(), plane,
            region_min_xy_m=(-.04, -.04), region_max_xy_m=(.04, .04))


def test_dense_footprint_coverage_rejects_extrapolation_and_coarse_grid():
    cut = capture()
    plane = project_depth_mask_support_surface(
        cut, mask(), mask_clipped=False, maximum_depth_m=2.)
    with pytest.raises(ValueError, match="outside the observed"):
        check_observed_support_footprint(
            cut, mask(), plane, region_min_xy_m=(-.09, -.04),
            region_max_xy_m=(.04, .04))
    with pytest.raises(ValueError, match="qualification profile"):
        check_observed_support_footprint(
            cut, mask(), plane, region_min_xy_m=(-.04, -.04),
            region_max_xy_m=(.04, .04), resolution_m=.01)
    with pytest.raises(ValueError, match="different source cuts"):
        check_observed_support_footprint(
            cut, mask(), replace(plane, capture_group_id="other-cut"),
            region_min_xy_m=(-.04, -.04), region_max_xy_m=(.04, .04))
    with pytest.raises(ValueError, match="bounded sample budget"):
        check_observed_support_footprint(
            cut, mask(), replace(plane, observed_bounds_max_xy_m=(1e12, 1e12)),
            region_min_xy_m=(0., 0.), region_max_xy_m=(1e12, 1e12))


def known_support_for_cut():
    transform = np.eye(4)
    transform[:3, 3] = (0., 0., -.02)
    return CalibratedBoxSupportObservation(
        support_id="table", reference_frame_id="world", scene_revision=3,
        measured_at=10., reference_from_box=transform,
        box_size_m=(.12, .12, .04),
        geometry_evidence_ref="catalog:table", pose_evidence_ref="pose:table",
        geometry_complete=True,
    )


def test_depth_candidate_binds_known_support_only_with_matching_semantic_cut():
    cut = capture()
    search = search_depth_mask_support_surface(
        cut, mask(), mask_clipped=False, maximum_depth_m=2.,
        patch_size_px=16, stride_px=2)
    result = bind_known_box_support_from_depth(
        cut, search, known_support_for_cut(), semantic_support_id="table",
        expected_reference_frame_id="world",
        expected_support_mask_sha256=search.support_mask_sha256,
        lateral_uncertainty_m=.005, vertical_uncertainty_m=.002)
    assert result.support_id == "table"
    assert result.geometry_complete is True
    assert result.top_z_m == pytest.approx(.004)
    assert result.region_max_xy_m == pytest.approx((.055, .055))


@pytest.mark.parametrize("change, reason", [
    ({"semantic_support_id": "other"}, "Semantic support"),
    ({"expected_reference_frame_id": "camera"}, "Semantic support"),
    ({"expected_support_mask_sha256": "f" * 64}, "mask differs"),
    ({"expected_support_mask_sha256": True}, "mask differs"),
    ({"box": "stale"}, "correlated cut"),
    ({"search": "wrong-cut"}, "bound depth cut"),
])
def test_depth_box_fusion_rejects_identity_or_time_mismatch(change, reason):
    cut = capture()
    search = search_depth_mask_support_surface(
        cut, mask(), mask_clipped=False, maximum_depth_m=2.,
        patch_size_px=16, stride_px=2)
    box = known_support_for_cut()
    if change.get("box") == "stale":
        box = replace(box, measured_at=9.)
    if change.get("search") == "wrong-cut":
        search = replace(search, candidate=replace(search.candidate, capture_group_id="other"))
    values = dict(semantic_support_id="table", expected_reference_frame_id="world",
                  expected_support_mask_sha256=search.support_mask_sha256,
                  lateral_uncertainty_m=.005, vertical_uncertainty_m=.002)
    values.update({key: value for key, value in change.items() if key not in {"box", "search"}})
    with pytest.raises(ValueError, match=reason):
        bind_known_box_support_from_depth(cut, search, box, **values)

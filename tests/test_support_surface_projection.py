"""Measured depth support geometry is a proposal, never a motion permit."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from agent_gateway.profiles.back_projection import (
    LiveAlignedDepthCapture, project_depth_mask_support_surface,
    search_depth_mask_support_surface,
)


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

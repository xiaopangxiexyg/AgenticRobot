import json

import numpy as np
import pytest

from deployments.perception.cable_diagnostics import (
    cable_depth_evidence,
    cable_prompt_from_geometry,
    separate_thin_cable_mask,
)
from scripts.probes.inspect_grasp_cable import main as inspect_main


def test_metric_opening_separates_thin_strand_from_rigid_housing_without_mutating_inputs():
    mask = np.zeros((60, 90), bool)
    mask[8:42, 5:39] = True
    mask[23:28, 39:83] = True
    protected = np.zeros_like(mask); protected[24, 70] = True
    depth = np.full(mask.shape, .5)
    saved_mask, saved_depth = mask.copy(), depth.copy()
    result = separate_thin_cable_mask(depth_m=depth, mask=mask,
        protected_mask=protected, fx=400., maximum_diameter_m=.02)
    assert result.opening_radius_px == pytest.approx(8.)
    assert result.thin_mask[25, 60]
    assert not result.thin_mask[25, 20]
    assert result.thick_mask[25, 20]
    assert not result.thin_mask[24, 70] and not result.thick_mask[24, 70]
    np.testing.assert_array_equal(mask, saved_mask)
    np.testing.assert_array_equal(depth, saved_depth)


def test_metric_scale_changes_with_focal_length_and_distance():
    mask = np.ones((10, 10), bool)
    protected = np.zeros_like(mask)
    radii = [separate_thin_cable_mask(depth_m=np.full(mask.shape, z), mask=mask,
        protected_mask=protected, fx=fx, maximum_diameter_m=.02).opening_radius_px
        for fx, z in ((400., .5), (800., .5), (400., 1.))]
    assert radii == pytest.approx([8., 16., 4.])


@pytest.mark.parametrize("empty_depth", [False, True])
def test_missing_semantic_or_depth_support_does_not_invent_a_strand(empty_depth):
    mask = np.full((8, 8), empty_depth, bool)
    depth = np.zeros(mask.shape) if empty_depth else np.ones(mask.shape)
    result = separate_thin_cable_mask(depth_m=depth, mask=mask,
        protected_mask=np.zeros_like(mask), fx=400.)
    assert not result.thin_mask.any() and not result.thick_mask.any()
    assert result.opening_radius_px is None


def test_prompt_uses_connected_depth_support_not_silhouette_and_protects_target():
    mask = np.zeros((20, 30), bool); mask[8:11, 8:22] = True
    depth = np.ones(mask.shape)
    attachment = np.full(mask.shape, np.inf); attachment[8:11, 8:10] = .02
    rigid = np.full(mask.shape, np.inf); rigid[2:6, 2:6] = .005
    protected = np.zeros_like(mask); protected[2:6, 2:6] = True
    result = cable_prompt_from_geometry(depth_m=depth, thin_mask=mask,
        protected_mask=protected, attachment_distance_m=attachment, rigid_surface_distance_m=rigid)
    assert result is not None
    u, v = result.positive_uv
    assert mask[v, u] and not protected[v, u]
    assert result.negative_uv is None
    assert result.attachment_support_pixels == 6
    protected[:] = False
    result = cable_prompt_from_geometry(depth_m=depth, thin_mask=mask,
        protected_mask=protected, attachment_distance_m=attachment, rigid_surface_distance_m=rigid)
    u, v = result.negative_uv
    assert rigid[v, u] <= .008 and not mask[v, u]


def test_prompt_rejects_a_projected_strand_without_metric_attachment_support():
    mask = np.ones((12, 12), bool)
    assert cable_prompt_from_geometry(depth_m=np.ones(mask.shape), thin_mask=mask,
        protected_mask=np.zeros_like(mask), attachment_distance_m=np.full(mask.shape, .2),
        rigid_surface_distance_m=np.full(mask.shape, np.inf)) is None


def test_prompt_rejects_a_boolean_silhouette_as_a_metric_surface_distance():
    mask = np.ones((12, 12), bool)
    with pytest.raises(ValueError, match="Surface distances"):
        cable_prompt_from_geometry(depth_m=np.ones(mask.shape), thin_mask=mask,
            protected_mask=np.zeros_like(mask), attachment_distance_m=mask,
            rigid_surface_distance_m=np.full(mask.shape, np.inf))


def test_exact_depth_edge_search_matches_exhaustive_disk_and_keeps_inputs():
    rng = np.random.default_rng(20261015)
    depth = rng.uniform(.7, .8, (13, 17)).astype(np.float32)
    mask = np.zeros(depth.shape, bool); mask[3:10, 4:13] = rng.random((7, 9)) > .5
    protected = rng.random(depth.shape) < .08
    depth[8, 12] = 0
    original = depth.copy()
    result = cable_depth_evidence(depth_m=depth, mask=mask, protected_mask=protected,
        boundary_radius_px=3, depth_tolerance_m=.015)
    semantic = mask & ~protected
    reference_y, reference_x = np.nonzero(semantic & (depth > 0))
    for y, x in zip(*np.nonzero(~semantic & ~protected)):
        near = (reference_y-y)**2+(reference_x-x)**2 <= 9
        matches = bool(depth[y, x] > 0 and np.any(
            np.abs(depth[reference_y[near], reference_x[near]].astype(float)-float(depth[y, x])) <= .015))
        assert bool(result.depth_continuous_boundary[y, x]) is matches
    np.testing.assert_array_equal(depth, original)


def test_depth_edge_search_does_not_assume_values_between_neighbour_depths_exist():
    mask = np.zeros((7, 7), bool); mask[3, 2] = True; mask[3, 4] = True
    depth = np.ones(mask.shape); depth[3, 2] = .7; depth[3, 4] = .9; depth[3, 3] = .8
    protected = np.zeros_like(mask)
    result = cable_depth_evidence(depth_m=depth, mask=mask, protected_mask=protected,
        boundary_radius_px=1, depth_tolerance_m=.015)
    assert result.depth_unexplained_boundary[3, 3]
    assert not result.depth_continuous_boundary[3, 3]


def test_boundary_classes_are_disjoint_and_foreground_target_is_not_erased():
    mask = np.zeros((9, 9), bool); mask[4, 4] = True
    depth = np.full(mask.shape, .75); depth[4, 5] = .77; depth[5, 4] = 0
    protected = np.zeros_like(mask); protected[4, 3] = True
    result = cable_depth_evidence(depth_m=depth, mask=mask, protected_mask=protected,
        boundary_radius_px=1)
    assert result.depth_unexplained_boundary[4, 5]
    assert result.missing_depth_boundary[5, 4]
    assert result.depth_continuous_boundary[3, 4]
    for field in (result.semantic_mask, result.depth_continuous_boundary,
                  result.depth_unexplained_boundary, result.missing_depth_boundary):
        assert not (field & protected).any()
    total = sum(field.astype(int) for field in (result.depth_continuous_boundary,
        result.depth_unexplained_boundary, result.missing_depth_boundary))
    assert total.max() <= 1
    assert depth[4, 5] == .77 and depth[5, 4] == 0


def test_mask_frame_border_does_not_wrap_to_the_other_edge():
    mask = np.zeros((5, 5), bool); mask[0, 0] = True
    result = cable_depth_evidence(depth_m=np.ones(mask.shape), mask=mask,
        protected_mask=np.zeros_like(mask), boundary_radius_px=1)
    assert result.depth_continuous_boundary[0, 1]
    assert not result.depth_continuous_boundary[0, 4]


def test_missing_reference_depth_is_reported_as_unexplained_not_safe():
    mask = np.zeros((5, 5), bool); mask[2, 2] = True
    depth = np.ones(mask.shape); depth[2, 2] = 0
    result = cable_depth_evidence(depth_m=depth, mask=mask,
        protected_mask=np.zeros_like(mask), boundary_radius_px=1)
    assert not result.depth_continuous_boundary.any()
    assert result.depth_unexplained_boundary[2, 3]


@pytest.mark.parametrize("kwargs", [{"boundary_radius_px": -1}, {"boundary_radius_px": 17},
    {"boundary_radius_px": True}, {"depth_tolerance_m": 0}, {"depth_tolerance_m": float("nan")}])
def test_invalid_diagnostic_parameters_fail_at_the_array_boundary(kwargs):
    with pytest.raises(ValueError):
        cable_depth_evidence(depth_m=np.ones((2, 2)), mask=np.zeros((2, 2), bool),
            protected_mask=np.zeros((2, 2), bool), **kwargs)


def test_probe_keeps_source_arrays_and_outputs_non_authoritative_diagnostics(tmp_path, capsys):
    depth = np.ones((30, 40), np.float32)
    target = np.zeros(depth.shape, bool); target[15, 15] = True
    mask = np.zeros_like(target); mask[12:17, 5:35] = True
    rgb = np.zeros((*depth.shape, 3), np.uint8)
    snapshot, masks, output = tmp_path/"snapshot.npz", tmp_path/"masks.npz", tmp_path/"output"
    np.savez(snapshot, depth_m=depth, target_mask=target, rgb=rgb)
    np.savez(masks, cable=mask)
    original_bytes = snapshot.read_bytes()
    assert inspect_main(["--snapshot", str(snapshot), "--mask-npz", str(masks), "--mask-key", "cable",
                         "--fx", "400", "--output", str(output)]) == 0
    report = json.loads((output/"report.json").read_text())
    assert report["background_points_deleted"] == report["target_points_deleted"] == 0
    assert report["devices_started"] is False
    assert report["target_protection_available"] is True
    assert report["prompt_status"] == "not_requested"
    assert snapshot.read_bytes() == original_bytes
    assert (output/"overlay.png").is_file()
    with pytest.raises(FileExistsError):
        inspect_main(["--snapshot", str(snapshot), "--mask-npz", str(masks), "--mask-key", "cable",
                      "--fx", "400", "--output", str(output)])
    assert "offline_diagnostic" in capsys.readouterr().out


def test_probe_uses_recorded_intrinsics_and_explicit_robot_distance_hints(tmp_path):
    depth = np.ones((30, 40), np.float32)
    mask = np.zeros(depth.shape, bool); mask[12:15, 10:30] = True
    protected = np.zeros_like(mask)
    attachment = np.full(depth.shape, np.inf); attachment[12:15, 10:12] = .01
    rigid = np.full(depth.shape, np.inf); rigid[4:8, 4:8] = .005
    metadata = np.frombuffer(json.dumps({"capture": {"camera_intrinsics": {"fx": 400.}}}).encode(), np.uint8)
    snapshot, masks, distances = (tmp_path/name for name in ("input.npz", "masks.npz", "distances.npz"))
    np.savez(snapshot, depth_m=depth, rgb=np.zeros((*depth.shape, 3), np.uint8),
        target_mask=protected, metadata=metadata)
    np.savez(masks, cable=mask)
    np.savez(distances, attachment_distance_m=attachment, rigid_surface_distance_m=rigid)
    output = tmp_path/"diagnostic"
    inspect_main(["--snapshot", str(snapshot), "--mask-npz", str(masks), "--mask-key", "cable",
                  "--robot-distances", str(distances), "--output", str(output)])
    report = json.loads((output/"report.json").read_text())
    assert report["prompt_status"] == "suggested"
    positive, negative = report["prompt"]["positive_uv"], report["prompt"]["negative_uv"]
    assert mask[positive[1], positive[0]]
    assert rigid[negative[1], negative[0]] <= .008
    assert report["opening_radius_px"] == pytest.approx(5.)


@pytest.mark.parametrize("larger_state", ["rigid", "no_depth", "protected", "valid"])
def test_prompt_ranks_only_components_with_an_eligible_positive_seed(larger_state):
    thin = np.zeros((24, 32), dtype=bool)
    larger = np.zeros_like(thin)
    smaller = np.zeros_like(thin)
    larger[3:8, 3:12] = True
    smaller[14:17, 22:28] = True
    thin |= larger | smaller
    depth = np.ones(thin.shape, dtype=np.float32)
    protected = np.zeros_like(thin)
    attachment = np.full(thin.shape, np.inf)
    attachment[thin] = .01
    rigid = np.full(thin.shape, np.inf)
    if larger_state == "rigid":
        rigid[larger] = .005
    elif larger_state == "no_depth":
        depth[larger] = 0
    elif larger_state == "protected":
        protected[larger] = True
    originals = [value.copy() for value in (depth, thin, protected, attachment, rigid)]
    result = cable_prompt_from_geometry(depth_m=depth, thin_mask=thin,
        protected_mask=protected, attachment_distance_m=attachment,
        rigid_surface_distance_m=rigid)
    assert result is not None, "A supported component with a valid positive seed must not be discarded"
    expected = larger if larger_state == "valid" else smaller
    u, v = result.positive_uv
    assert expected[v, u] and not protected[v, u]
    assert depth[v, u] > 0 and rigid[v, u] > .008
    assert result.component_pixels == int(expected.sum())
    assert result.attachment_support_pixels == int(expected.sum())
    for actual, original in zip((depth, thin, protected, attachment, rigid), originals):
        np.testing.assert_array_equal(actual, original)

"""Same-cut mask selection reusing the registered RGB-D and geometry owners."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from agent_gateway.profiles.back_projection import (
    CorrelatedPlacementCapture,
    project_depth_mask_geometry,
)
from agent_gateway.profiles.segmentation import decode_segmentation_mask_npy
from agent_gateway.skills.perception.surface_anchor_v2 import (
    EXACT_ALGORITHM,
    INTERIOR_ALGORITHM,
    MAX_CLEARANCE_MEDOID_ALGORITHM,
    SELECTION_FIELDS,
    validate_interior_algorithm,
)


def select_mask_interior_pixel(
    mask, depth_m, *, requested_pixel_uv, maximum_depth_m,
    interior_algorithm=INTERIOR_ALGORITHM,
):
    """Select an actual max-clearance pixel using the registered tie policy.

    Image borders count as boundaries. For a point prompt only its connected
    mask component may supply a point; missing depth never creates a new point.
    The default keeps prompt-nearest ties. The optional medoid policy minimizes
    total image-space L1 distance within the max-clearance candidates only.
    """
    from scipy.ndimage import distance_transform_edt, label

    validate_interior_algorithm(interior_algorithm)
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m)
    if mask.ndim != 2 or mask.shape != depth.shape or not mask.any():
        raise ValueError("Surface selection requires a nonempty aligned mask.")
    eligible = mask
    rows, cols = np.nonzero(mask)
    center = np.array([np.median(rows), np.median(cols)])
    if requested_pixel_uv is not None:
        u, v = requested_pixel_uv
        if not (0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]) or not mask[v, u]:
            raise RuntimeError("surface_anchor_prompt_not_in_mask")
        components, _ = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        eligible = components == components[v, u]
        center = np.array([v, u])
    inward = distance_transform_edt(np.pad(eligible, 1))[1:-1, 1:-1]
    valid = eligible & np.isfinite(depth) & (depth > 0) & (depth <= maximum_depth_m)
    if not valid.any():
        raise RuntimeError("surface_anchor_mask_has_no_valid_depth")
    maximum = float(inward[valid].max())
    candidates = np.argwhere(valid & (inward == maximum))
    if interior_algorithm == MAX_CLEARANCE_MEDOID_ALGORITHM:
        cost = np.zeros(len(candidates), dtype=np.int64)
        for axis in (0, 1):
            values = candidates[:, axis]
            ordered = np.sort(values)
            prefix = np.concatenate(([0], np.cumsum(ordered, dtype=np.int64)))
            left = np.searchsorted(ordered, values, side="left")
            # Exact sum of absolute distances in O(k log k), not a k-by-k matrix.
            cost += values * left - prefix[left]
            cost += prefix[-1] - prefix[left] - values * (len(values) - left)
    else:
        cost = np.sum((candidates - center) ** 2, axis=1)
    # argwhere is row-major, so argmin provides deterministic final ties.
    v, u = candidates[int(np.argmin(cost))]
    return [int(u), int(v)], maximum


def capture_selected_surface(
    *, provider, segmentation_provider, parameters, expected_station_revision,
    interior_algorithm=INTERIOR_ALGORITHM,
):
    """Segment and project one already correlated cut, without a second read."""
    cut = provider.capture_port.capture_placement_rgbd(
        role=parameters["role"], expected_station_revision=expected_station_revision
    )
    if not isinstance(cut, CorrelatedPlacementCapture):
        raise TypeError("Surface anchor requires a typed correlated RGB-D cut.")
    if cut.rgb.captured_at != cut.depth.captured_at:
        raise RuntimeError(
            "Surface anchor RGB and depth must belong to the same capture time."
        )
    selected = parameters["target_pixel_uv"]
    selection = dict.fromkeys(SELECTION_FIELDS)
    selection.update(
        mode=parameters["selection_mode"],
        requested_pixel_uv=selected,
        language_prompt=parameters["language_prompt"],
        min_score=parameters["min_score"],
        algorithm_id=EXACT_ALGORITHM,
        input_rgb_content_hash=cut.rgb.rgb_content_hash,
    )
    prepared = None
    geometry = None
    if parameters["selection_mode"] == "mask_interior":
        segment_params = {
            key: parameters[key]
            for key in ("role", "target_pixel_uv", "language_prompt", "min_score")
        }
        prepared = segmentation_provider.segment_capture(
            capture=cut.rgb, parameters=segment_params
        )
        evidence = prepared.execution_result
        mask = decode_segmentation_mask_npy(
            prepared.mask_npy,
            width=cut.rgb.image.width,
            height=cut.rgb.image.height,
            expected_count=evidence["mask_pixel_count"],
            expected_bounds=tuple(evidence["mask_bounds_xyxy"]),
        )
        selected, distance = select_mask_interior_pixel(
            mask,
            cut.depth.depth_m,
            requested_pixel_uv=parameters["target_pixel_uv"],
            maximum_depth_m=provider.registration.profile.maximum_depth_m,
            interior_algorithm=interior_algorithm,
        )
        profile = provider.registration.profile
        geometry = asdict(
            project_depth_mask_geometry(
                cut.depth,
                mask,
                minimum_valid_points=1,
                maximum_depth_m=profile.maximum_depth_m,
                robust_lower_quantile=profile.robust_lower_quantile,
                robust_upper_quantile=profile.robust_upper_quantile,
            )
        )
        selection.update(
            {
                key: evidence[key]
                for key in (
                    "mask_media_content_hash",
                    "overlay_media_content_hash",
                    "model_profile_id",
                    "model_profile_hash",
                    "mask_bounds_xyxy",
                    "mask_pixel_count",
                    "score",
                )
            }
        )
        selection.update(algorithm_id=interior_algorithm, interior_distance_px=distance)
    measurement = provider.project_capture(
        cut.depth,
        {
            "role": parameters["role"],
            "target_pixel_uv": selected,
            "pixel_region_xyxy": None,
            "world_z_min_m": None,
            "world_z_max_m": None,
        },
        expected_station_revision=expected_station_revision,
    )
    return (
        measurement,
        cut.depth,
        {
            "selected_pixel_uv": selected,
            "selection": selection,
            "mask_geometry": geometry,
        },
        prepared,
    )

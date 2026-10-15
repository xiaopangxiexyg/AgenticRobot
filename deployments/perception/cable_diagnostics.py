"""Non-authoritative cable prompts and aligned RGB-D boundary diagnostics.

These functions never change depth, target labels or a planning scene. Proximity
to a robot surface is a prompt hint, not proof that a cable belongs to that body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label


def _aligned_inputs(depth_m, mask, protected_mask):
    depth = np.asarray(depth_m)
    mask = np.asarray(mask)
    protected = np.asarray(protected_mask)
    if (depth.ndim != 2 or mask.shape != depth.shape or protected.shape != depth.shape
            or mask.dtype != np.bool_ or protected.dtype != np.bool_
            or not np.issubdtype(depth.dtype, np.floating)
            or not np.all(np.isfinite(depth)) or np.any(depth < 0)):
        raise ValueError("Expected aligned finite nonnegative depth and boolean masks")
    return depth, mask, protected


def _positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _roi(mask, padding=0):
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return (slice(max(0, rows.min()-padding), min(mask.shape[0], rows.max()+padding+1)),
            slice(max(0, columns.min()-padding), min(mask.shape[1], columns.max()+padding+1)))


@dataclass(frozen=True)
class ThinCableMask:
    thin_mask: np.ndarray
    thick_mask: np.ndarray
    opening_radius_px: float | None


def separate_thin_cable_mask(*, depth_m: np.ndarray, mask: np.ndarray,
                             protected_mask: np.ndarray, fx: float,
                             maximum_diameter_m: float = .025) -> ThinCableMask:
    """Metric-scaled opening/top-hat, not a fixed 9-pixel removal rule.

    This approximation uses the median valid depth of the proposal. It is not a
    diameter measurement or an ownership decision, especially across occlusions.
    """
    depth, original, protected = _aligned_inputs(depth_m, mask, protected_mask)
    _positive(fx, "fx")
    _positive(maximum_diameter_m, "maximum_diameter_m")
    eligible = original & ~protected
    valid = eligible & (depth > 0)
    thin = np.zeros(original.shape, dtype=bool)
    thick = np.zeros_like(thin)
    region = _roi(eligible)
    if region is None or not np.any(valid):
        return ThinCableMask(thin, thick, None)
    radius = float(fx * maximum_diameter_m / (2*float(np.median(depth[valid]))))
    local = eligible[region]
    inward = distance_transform_edt(np.pad(local, 1))[1:-1, 1:-1]
    core = inward > radius
    if np.any(core):
        opened = (distance_transform_edt(~core) <= radius) & local
        thick[region] = opened
        thin[region] = local & ~opened
    else:
        thin[region] = local
    return ThinCableMask(thin, thick, radius)


@dataclass(frozen=True)
class CablePrompt:
    positive_uv: tuple[int, int]
    negative_uv: tuple[int, int] | None
    component_pixels: int
    attachment_support_pixels: int


def cable_prompt_from_geometry(*, depth_m: np.ndarray, thin_mask: np.ndarray,
                               protected_mask: np.ndarray,
                               attachment_distance_m: np.ndarray,
                               rigid_surface_distance_m: np.ndarray,
                               attachment_tolerance_m: float = .03,
                               rigid_tolerance_m: float = .008,
                               minimum_support_pixels: int = 3) -> CablePrompt | None:
    """Choose a thin connected proposal near a body; use rigid depth for negatives.

    Distances must describe the same captured frame in meters. Finger-state
    unions can be attachment hints, but must not be supplied as measured rigid
    surfaces. A 2-D robot silhouette alone cannot justify a negative prompt.
    """
    depth, thin, protected = _aligned_inputs(depth_m, thin_mask, protected_mask)
    _positive(attachment_tolerance_m, "attachment_tolerance_m")
    _positive(rigid_tolerance_m, "rigid_tolerance_m")
    if (isinstance(minimum_support_pixels, bool)
            or not isinstance(minimum_support_pixels, int) or minimum_support_pixels < 1):
        raise ValueError("minimum_support_pixels must be a positive integer")
    distances = []
    for value in (attachment_distance_m, rigid_surface_distance_m):
        value = np.asarray(value)
        if (value.shape != depth.shape or not np.issubdtype(value.dtype, np.floating)
                or np.any(np.isnan(value)) or np.any(value < 0)):
            raise ValueError("Surface distances must be aligned nonnegative meters or +inf")
        distances.append(value)
    attachment, rigid = distances
    eligible = thin & ~protected
    components, _ = label(eligible, structure=np.ones((3, 3), dtype=bool))
    selected = None
    best_rank = None
    for component, region in enumerate(find_objects(components), start=1):
        if region is None:
            continue
        local = components[region] == component
        support = local & (depth[region] > 0) & (attachment[region] <= attachment_tolerance_m)
        count = int(support.sum())
        if count < minimum_support_pixels:
            continue
        # Prefer a substantial connected strand, not an arbitrary top-score tie.
        rank = (int(local.sum()), count)
        if best_rank is None or rank > best_rank:
            selected, best_rank = (component, region, count), rank
    if selected is None:
        return None
    component, region, count = selected
    local = components[region] == component
    inward = distance_transform_edt(np.pad(local, 1))[1:-1, 1:-1]
    eligible_seed = local & (depth[region] > 0) & (rigid[region] > rigid_tolerance_m)
    if not np.any(eligible_seed):
        return None
    # A strand-interior point avoids placing the positive prompt on its seam.
    score = np.where(eligible_seed, inward, -1.)
    row, column = np.unravel_index(int(np.argmax(score)), score.shape)
    positive = (int(column+region[1].start), int(row+region[0].start))
    negative = None
    rigid_pixels = (depth > 0) & (rigid <= rigid_tolerance_m) & ~protected & ~thin
    negative_region = _roi(rigid_pixels)
    if negative_region is not None:
        inside = distance_transform_edt(np.pad(rigid_pixels[negative_region], 1))[1:-1, 1:-1]
        row, column = np.unravel_index(int(np.argmax(inside)), inside.shape)
        negative = (int(column+negative_region[1].start), int(row+negative_region[0].start))
    return CablePrompt(positive, negative, int(local.sum()), count)


@dataclass(frozen=True)
class CableDepthEvidence:
    semantic_mask: np.ndarray
    depth_continuous_boundary: np.ndarray
    depth_unexplained_boundary: np.ndarray
    missing_depth_boundary: np.ndarray
    protected_overlap: np.ndarray


def cable_depth_evidence(*, depth_m: np.ndarray, mask: np.ndarray,
                         protected_mask: np.ndarray, boundary_radius_px: int = 4,
                         depth_tolerance_m: float = .015) -> CableDepthEvidence:
    """Classify an RGB boundary with an exact local depth search inside its ROI.

    Even depth-continuous pixels are not confirmed self geometry. All outputs
    are diagnostic labels; no missing/inconsistent sample becomes free space.
    """
    depth, original, protected = _aligned_inputs(depth_m, mask, protected_mask)
    if (isinstance(boundary_radius_px, bool) or not isinstance(boundary_radius_px, int)
            or not 0 <= boundary_radius_px <= 16):
        raise ValueError("boundary_radius_px must be an integer in [0, 16]")
    _positive(depth_tolerance_m, "depth_tolerance_m")
    semantic = original & ~protected
    continuous = np.zeros(depth.shape, dtype=bool)
    unexplained = np.zeros_like(continuous)
    missing = np.zeros_like(continuous)
    region = _roi(semantic, boundary_radius_px)
    if region is not None:
        radius = boundary_radius_px
        local = semantic[region]
        # Padding is explicit: frame borders must not wrap or synthesize depths.
        reference = np.pad(local & (depth[region] > 0), radius)
        z = np.pad(depth[region].astype(float), radius)
        observed = depth[region].astype(float)
        valid = observed > 0
        supported = np.zeros(local.shape, dtype=bool)
        h, w = local.shape
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if dx*dx+dy*dy > radius*radius:
                    continue
                ys, xs = slice(radius+dy, radius+dy+h), slice(radius+dx, radius+dx+w)
                reference_here = reference[ys, xs]
                supported |= reference_here & valid & (np.abs(observed-z[ys, xs]) <= depth_tolerance_m)
        # The band includes semantic pixels without depth, but never target pixels.
        band = (distance_transform_edt(~local) <= radius) & ~local & ~protected[region]
        continuous[region] = band & supported
        unexplained[region] = band & valid & ~supported
        missing[region] = band & ~valid
    return CableDepthEvidence(semantic, continuous, unexplained, missing, original & protected)

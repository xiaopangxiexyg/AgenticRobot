"""Target-associated RGB-D fusion, independent of devices and motion authority."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from robot_motion.grasp_visible_scene import (
    PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER,
    ParallelJawSolidGeometry,
    VisibleGraspScene,
)


@dataclass(frozen=True)
class GraspRgbdView:
    role: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: Mapping[str, float]
    reference_from_camera: np.ndarray
    acquired_at: float


@dataclass(frozen=True)
class GraspViewFusionPolicy:
    voxel_size_m: float = 0.004
    depth_agreement_m: float = 0.015
    max_frame_skew_sec: float = 0.2
    min_target_points: int = 32
    target_component_radius_m: float = 0.014
    minimum_seed_component_fraction: float = 0.8

    def __post_init__(self) -> None:
        for value in (self.voxel_size_m, self.depth_agreement_m, self.max_frame_skew_sec, self.target_component_radius_m):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Fusion distances and time bounds must be positive")
        if self.min_target_points < 3:
            raise ValueError("Fusion requires at least three target points")
        if not 0.5 < self.minimum_seed_component_fraction <= 1:
            raise ValueError("Seed component must dominate the observed target")


@dataclass(frozen=True)
class FusedGraspObservation:
    scene: VisibleGraspScene
    target_masks: Mapping[str, np.ndarray]
    view_diagnostics: tuple[dict, ...]


def _validate_view(view: GraspRgbdView) -> None:
    depth = view.depth_m
    transform = np.asarray(view.reference_from_camera)
    if (depth.ndim != 2 or view.rgb.shape != (*depth.shape, 3)
            or view.rgb.dtype != np.uint8 or not np.all(np.isfinite(depth))
            or np.any(depth < 0) or not np.isfinite(view.acquired_at)):
        raise ValueError("Invalid correlated RGB-D view")
    if (transform.shape != (4, 4) or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-7)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6)):
        raise ValueError("RGB-D view requires a rigid camera transform")
    if any(not np.isfinite(float(view.intrinsics[k])) for k in ("fx", "fy", "cx", "cy")):
        raise ValueError("Invalid RGB-D intrinsics")
    if min(view.intrinsics["fx"], view.intrinsics["fy"]) <= 0:
        raise ValueError("Invalid RGB-D focal length")


def _points(view: GraspRgbdView, mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask & (view.depth_m > 0))
    z = view.depth_m[rows, cols]
    k = view.intrinsics
    xyz = np.column_stack(((cols-k["cx"])*z/k["fx"], (rows-k["cy"])*z/k["fy"], z))
    pose = view.reference_from_camera
    return xyz @ pose[:3, :3].T + pose[:3, 3]


def _voxel_indices(points: np.ndarray, size: float) -> np.ndarray:
    if not len(points):
        return np.empty(0, dtype=np.int64)
    _, indices = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return np.sort(indices)


def refine_seed_target_mask(view: GraspRgbdView, mask: np.ndarray, pixel_uv: tuple[int, int],
                            policy: GraspViewFusionPolicy) -> tuple[np.ndarray, dict]:
    """Separate disconnected mask-depth islands, without deleting any geometry.

    RGB segmentation can include background depth at an occlusion boundary.
    Keep only a dominant, click-connected metric component as the target. If
    evidence is split, require another observation instead of picking a blob.
    The caller must retain rejected points as non-contact background obstacles.
    """
    u, v = pixel_uv
    if not (0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]) or not mask[v, u]:
        raise ValueError("Target seed must be inside its segmentation mask")
    rows, cols = np.nonzero(mask & (view.depth_m > 0))
    if len(rows) < policy.min_target_points:
        raise RuntimeError("Anchor view has insufficient metric target evidence")
    pixel_distance = (rows-v)**2 + (cols-u)**2
    seed_index = int(np.argmin(pixel_distance))
    if pixel_distance[seed_index] > 25:
        raise RuntimeError("Target seed has no local valid depth")
    points = _points(view, mask)
    voxels = points[_voxel_indices(points, policy.voxel_size_m)]
    tree = cKDTree(voxels)
    pairs = tree.query_pairs(policy.target_component_radius_m, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(voxels), len(voxels)))
    component_count, labels = connected_components(graph, directed=False)
    _, nearest = tree.query(points)
    selected_label = labels[nearest[seed_index]]
    selected = labels[nearest] == selected_label
    fraction = float(np.mean(labels == selected_label))
    if fraction < policy.minimum_seed_component_fraction:
        raise RuntimeError(f"Target depth components are ambiguous: seed_fraction={fraction:.3f}")
    refined = mask.copy()
    refined[rows, cols] = selected
    return refined, {"method": "seed_connected_metric_target_v1", "component_count": int(component_count),
        "component_radius_m": policy.target_component_radius_m, "seed_component_voxel_fraction": fraction,
        "input_target_depth_points": len(points), "retained_target_depth_points": int(selected.sum()),
        "relabelled_background_points": int((~selected).sum()), "deleted_scene_points": 0}


def _associate_target(
    view: GraspRgbdView, anchor_points: np.ndarray,
    segment_point: Callable, policy: GraspViewFusionPolicy,
    anchor: GraspRgbdView, anchor_mask: np.ndarray,
) -> tuple[np.ndarray | None, str, int, dict]:
    """Use visible anchor surface samples, not a possibly occluded 3-D centroid."""
    pose = view.reference_from_camera
    xyz = (anchor_points - pose[:3, 3]) @ pose[:3, :3]
    xyz = xyz[xyz[:, 2] > 0]
    k = view.intrinsics
    pixels = np.rint(xyz[:, :2] / xyz[:, 2:3] * (k["fx"], k["fy"]) + (k["cx"], k["cy"])).astype(int)
    h, w = view.depth_m.shape
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < w)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    pixels, xyz = pixels[inside], xyz[inside]
    if not len(pixels):
        return None, "target_out_of_view", 0, {}
    observed = view.depth_m[pixels[:, 1], pixels[:, 0]]
    matched = (observed > 0) & (np.abs(observed-xyz[:, 2]) <= policy.depth_agreement_m)
    if np.count_nonzero(matched) < 3:
        reason = "target_occluded" if np.all((observed > 0) & (observed < xyz[:, 2]-policy.depth_agreement_m)) else "target_depth_unconfirmed"
        return None, reason, 0, {}
    pixels = np.unique(pixels[matched], axis=0)
    if len(pixels) < 3:
        return None, "insufficient_shared_target_surface", len(pixels), {}
    center = np.median(pixels, axis=0)
    seed = pixels[np.argmin(np.sum((pixels-center)**2, axis=1))]
    mask = np.asarray(segment_point(Image.fromarray(view.rgb), pixel_uv=tuple(int(x) for x in seed)), dtype=bool)
    if mask.shape != view.depth_m.shape or not mask[seed[1], seed[0]]:
        raise RuntimeError("Auxiliary target segmenter returned an invalid mask")
    agreement = np.mean(mask[pixels[:, 1], pixels[:, 0]])
    target = _points(view, mask)
    evidence = {"target_depth_points": len(target),
                "shared_pixel_mask_agreement": float(agreement)}
    if len(target) < policy.min_target_points or agreement < 0.5:
        return None, "target_association_rejected", len(pixels), evidence
    # A mask containing the seed is insufficient: reject a disconnected or
    # grossly over-segmented object. This is an association check, not completion
    # of unseen geometry or permission to move through unobserved space.
    distance, _ = cKDTree(anchor_points).query(target)
    span = float(np.linalg.norm(np.ptp(anchor_points, axis=0)))
    distance_p95 = float(np.quantile(distance, 0.95))
    evidence.update(target_to_anchor_distance_p95_m=distance_p95,
        target_xyz_span_m=np.ptp(target, axis=0).tolist(),
        target_depth_quantiles_m=np.quantile(view.depth_m[mask & (view.depth_m > 0)], [.05, .5, .95]).tolist())
    if distance_p95 > max(0.10, 2.0 * span):
        return None, "target_geometry_inconsistent", len(pixels), evidence
    # Check the reverse association only where the anchor actually observes
    # that surface. Hidden backsides may add evidence; visible neighbours must
    # not become exempt target-contact geometry through an oversized mask.
    anchor_xyz = (target-anchor.reference_from_camera[:3, 3]) @ anchor.reference_from_camera[:3, :3]
    anchor_xyz = anchor_xyz[anchor_xyz[:, 2] > 0]
    intr = anchor.intrinsics
    uv = np.rint(anchor_xyz[:, :2]/anchor_xyz[:, 2:3]*(intr["fx"], intr["fy"])+(intr["cx"], intr["cy"])).astype(int)
    h, w = anchor.depth_m.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, anchor_xyz = uv[inside], anchor_xyz[inside]
    observed = anchor.depth_m[uv[:, 1], uv[:, 0]]
    confirmed = (observed > 0) & (np.abs(observed-anchor_xyz[:, 2]) <= policy.depth_agreement_m)
    conflicting = confirmed & ~anchor_mask[uv[:, 1], uv[:, 0]]
    confirmed_count, conflict_count = int(np.count_nonzero(confirmed)), int(np.count_nonzero(conflicting))
    # Evidence describes the existing gate; it never authorizes target contact
    # or deletes geometry. Count surface samples, as the original gate does.
    evidence.update(reverse_confirmed_surface_samples=confirmed_count,
        reverse_background_conflict_samples=conflict_count,
        reverse_background_conflict_fraction=conflict_count / max(1, confirmed_count))
    if conflict_count >= 3 and conflict_count > .1*confirmed_count:
        return None, "target_background_conflict", len(pixels), evidence
    return mask, "target_and_environment", len(pixels), evidence


def fuse_grasp_views(
    *, anchor: GraspRgbdView, anchor_mask: np.ndarray,
    auxiliary: Sequence[GraspRgbdView], geometry: ParallelJawSolidGeometry,
    segment_point: Callable,
    anchor_pixel_uv: tuple[int, int] | None = None,
    clean_depth: Callable[[GraspRgbdView, np.ndarray], np.ndarray] | None = None,
    policy: GraspViewFusionPolicy = GraspViewFusionPolicy(),
) -> FusedGraspObservation:
    """Fuse only associated views; returned points remain in one reference frame.

    Auxiliary failures never erase anchor evidence. Invalid poses and segmenter
    failures are errors; normal occlusion/depth failures are reported exclusions.
    A view with no confirmed target is not silently labelled all-background.
    """
    views = (anchor, *auxiliary)
    if not 1 <= len(views) <= 3 or len({v.role for v in views}) != len(views):
        raise ValueError("Fusion requires one to three distinct views")
    for view in views:
        _validate_view(view)
    mask = np.asarray(anchor_mask)
    if mask.dtype != np.bool_ or mask.shape != anchor.depth_m.shape:
        raise ValueError("Anchor mask does not match the captured RGB-D frame")
    original_anchor_mask = mask
    refinement = None
    if anchor_pixel_uv is not None:
        mask, refinement = refine_seed_target_mask(anchor, mask, anchor_pixel_uv, policy)
    anchor_points = _points(anchor, mask)
    if len(anchor_points) < policy.min_target_points:
        raise RuntimeError("Anchor view has insufficient metric target evidence")
    anchor_points = anchor_points[_voxel_indices(anchor_points, policy.voxel_size_m)]
    masks = {anchor.role: mask.copy()}
    diagnostics = []
    scenes = []
    for index, view in enumerate(views):
        association_evidence = {}
        if index == 0:
            selected_mask, reason, shared = mask, "anchor", len(anchor_points)
        elif abs(view.acquired_at-anchor.acquired_at) > policy.max_frame_skew_sec:
            diagnostics.append({"role": view.role, "used": False, "reason": "frame_skew_exceeded"})
            continue
        else:
            selected_mask, reason, shared, association_evidence = _associate_target(
                view, anchor_points, segment_point, policy, anchor, mask)
        evidence = {"association_evidence": association_evidence} if association_evidence else {}
        if selected_mask is None:
            diagnostics.append({"role": view.role, "used": False, "reason": reason,
                                "shared_target_pixels": shared, **evidence})
            continue
        masks[view.role] = selected_mask.copy()
        # Protect the original target pixels during self/background denoising.
        # Reclassifying a depth island must not cause its geometry to disappear.
        protected_mask = original_anchor_mask if index == 0 else selected_mask
        depth = view.depth_m if clean_depth is None else clean_depth(view, protected_mask)
        scene = VisibleGraspScene.from_depth(
            depth_m=depth, target_mask=protected_mask, intrinsics=view.intrinsics,
            reference_from_camera=view.reference_from_camera, geometry=geometry,
            depth_filter=PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER,
        )
        if index == 0 and refinement is not None:
            valid_original = original_anchor_mask & (view.depth_m > 0)
            if np.count_nonzero(scene.target_mask) != np.count_nonzero(valid_original):
                raise RuntimeError("Target refinement cannot drop protected depth geometry")
            labels = scene.target_mask.copy()
            labels[labels] = selected_mask[valid_original]
            scene = VisibleGraspScene(points=scene.points, target_mask=labels, geometry=geometry,
                                      depth_preprocessing_evidence=scene.depth_preprocessing_evidence)
        scenes.append(scene)
        diagnostics.append({"role": view.role, "used": True, "reason": reason,
                            "target_points": int(np.count_nonzero(scene.target_mask)),
                            "shared_target_pixels": shared, **evidence})
        if index == 0 and refinement is not None:
            diagnostics[-1]["target_depth_refinement"] = refinement
    points = np.concatenate([s.points for s in scenes])
    labels = np.concatenate([s.target_mask for s in scenes])
    # Keep target/background labels separate. Voxelisation cannot turn a
    # background obstacle into a target-contact exemption.
    retained = []
    for label in (False, True):
        indices = np.flatnonzero(labels == label)
        retained.extend(indices[_voxel_indices(points[indices], policy.voxel_size_m)])
    indices = np.asarray(retained, dtype=np.int64)
    return FusedGraspObservation(
        scene=VisibleGraspScene(points=points[indices], target_mask=labels[indices], geometry=geometry),
        target_masks=masks, view_diagnostics=tuple(diagnostics),
    )

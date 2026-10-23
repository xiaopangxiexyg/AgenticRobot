"""Visible-only parallel-jaw solid and sweep checks, independent of inference.

This is a conservative nominal proxy check, not contact mechanics, unseen-space
clearance, or whole-arm safety. The free space between fingers is never filled.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType

import numpy as np
from scipy.spatial import cKDTree

from dataset_contracts.canonical_encoding import canonical_content_hash

_IDENTITY_ROTATION_FLAT = (
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
)
# This only tapers the second-face preference at sub-pad feature scales. It is
# not a sensor accuracy bound or a minimum graspable object thickness.
_SURFACE_PAIR_RESOLUTION_PAD_FRACTION = 0.1


@dataclass(frozen=True)
class VisibleGraspDepthFilterConfig:
    """Conservative image-space rejection for isolated background depth spikes.

    The filter intentionally removes only background pixels that have no
    locally depth-consistent neighbour.  Target-labelled pixels are immutable,
    and connected surfaces remain untouched; this is not a general point-cloud
    smoother and cannot erase an obstacle merely because it is small.
    """

    neighborhood_radius_px: int = 1
    maximum_neighbor_depth_delta_m: float = 0.012
    minimum_supporting_neighbors: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.neighborhood_radius_px, bool)
            or not isinstance(self.neighborhood_radius_px, int)
            or not 1 <= self.neighborhood_radius_px <= 2
        ):
            raise ValueError("neighborhood_radius_px must be an integer in [1, 2]")
        if (
            not math.isfinite(self.maximum_neighbor_depth_delta_m)
            or not 0.0 < self.maximum_neighbor_depth_delta_m <= 0.05
        ):
            raise ValueError(
                "maximum_neighbor_depth_delta_m must be finite and in (0, 0.05]"
            )
        if (
            isinstance(self.minimum_supporting_neighbors, bool)
            or not isinstance(self.minimum_supporting_neighbors, int)
            or not 1 <= self.minimum_supporting_neighbors <= 8
        ):
            raise ValueError(
                "minimum_supporting_neighbors must be an integer in [1, 8]"
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            asdict(self),
            domain="dexteleop.motion.visible_grasp_depth_filter_config.v1",
        )


PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER = VisibleGraspDepthFilterConfig()


def filter_isolated_background_depth(
    *,
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    config: VisibleGraspDepthFilterConfig,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Return a valid-depth mask and hashable preprocessing evidence."""

    valid = depth_m > 0
    supporting = np.zeros(depth_m.shape, dtype=np.uint8)
    radius = config.neighborhood_radius_px
    height, width = depth_m.shape
    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            if row_offset == 0 and column_offset == 0:
                continue
            source_rows = slice(max(0, -row_offset), min(height, height - row_offset))
            source_columns = slice(
                max(0, -column_offset), min(width, width - column_offset)
            )
            neighbor_rows = slice(max(0, row_offset), min(height, height + row_offset))
            neighbor_columns = slice(
                max(0, column_offset), min(width, width + column_offset)
            )
            source_depth = depth_m[source_rows, source_columns]
            neighbor_depth = depth_m[neighbor_rows, neighbor_columns]
            consistent = (
                valid[source_rows, source_columns]
                & valid[neighbor_rows, neighbor_columns]
                & (
                    np.abs(source_depth - neighbor_depth)
                    <= config.maximum_neighbor_depth_delta_m
                )
            )
            supporting[source_rows, source_columns] += consistent.astype(np.uint8)
    remove = (
        valid
        & ~target_mask
        & (supporting < config.minimum_supporting_neighbors)
    )
    keep = valid & ~remove
    evidence = {
        "schema": "dexteleop.motion.visible_grasp_depth_preprocessing.v1",
        "method": "isolated_background_depth_spike_filter_v1",
        "input_valid_point_count": int(np.count_nonzero(valid)),
        "output_valid_point_count": int(np.count_nonzero(keep)),
        "removed_isolated_background_points": int(np.count_nonzero(remove)),
        "removed_target_points": 0,
        "config": asdict(config),
        "config_hash": config.semantic_content_hash,
    }
    return keep, evidence


@dataclass(frozen=True)
class ParallelJawSolidGeometry:
    palm_z_m: float
    finger_length_m: float
    finger_thickness_m: float
    pad_height_m: float
    maximum_opening_m: float
    tool_from_solid_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    solid_yaw_rad: float = 0.0
    tool_from_solid_rotation: tuple[float, ...] = _IDENTITY_ROTATION_FLAT

    def __post_init__(self) -> None:
        if not math.isfinite(self.palm_z_m):
            raise ValueError("palm_z_m must be finite")
        for value in (self.finger_length_m, self.finger_thickness_m,
                      self.pad_height_m, self.maximum_opening_m):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("gripper dimensions must be finite and positive")
        offset = np.asarray(self.tool_from_solid_xyz_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("solid frame offset must contain three finite meters")
        if not math.isfinite(self.solid_yaw_rad):
            raise ValueError("solid frame yaw must be finite")
        rotation = np.asarray(self.tool_from_solid_rotation, dtype=float)
        if rotation.size != 9 or not np.all(np.isfinite(rotation)):
            raise ValueError("solid frame rotation must contain nine finite values")
        rotation = rotation.reshape(3, 3)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.0):
            raise ValueError("solid frame rotation must be orthonormal")
        if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-7, rtol=0.0):
            raise ValueError("solid frame rotation must be right-handed")
        object.__setattr__(self, "tool_from_solid_xyz_m", tuple(float(x) for x in offset))
        object.__setattr__(
            self,
            "tool_from_solid_rotation",
            tuple(float(x) for x in rotation.reshape(-1)),
        )

    def solid_pose(self, tool_pose: np.ndarray) -> np.ndarray:
        """Locate +X-closing solids without changing the controller TCP.

        The optional full rotation is used for calibrated end-tool frames.  The
        legacy yaw remains additive for the original 4B2 proxy.  The analytic
        scene checks still receive the local solid pose, so a calibrated
        DMTAC model and the controller TCP cannot silently diverge.
        """
        transform = np.eye(4)
        c, s = math.cos(self.solid_yaw_rad), math.sin(self.solid_yaw_rad)
        yaw = np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=float)
        calibrated = np.asarray(self.tool_from_solid_rotation, dtype=float).reshape(3, 3)
        transform[:3, :3] = calibrated @ yaw
        transform[:3, 3] = self.tool_from_solid_xyz_m
        return np.asarray(tool_pose) @ transform


def _capture_points(
    local_target_points: np.ndarray, *, geometry: ParallelJawSolidGeometry,
    opening_m: float,
) -> np.ndarray:
    points = np.asarray(local_target_points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Capture support requires finite Nx3 model-frame points.")
    if (isinstance(opening_m, bool) or not math.isfinite(opening_m)
            or not 0 < opening_m <= geometry.maximum_opening_m):
        raise ValueError("Capture support opening must be within the geometry.")
    return points


def parallel_jaw_capture_support(
    local_target_points: np.ndarray, *, geometry: ParallelJawSolidGeometry,
    opening_m: float,
) -> tuple[int, float]:
    """Observe axial insertion, not transverse fit or contact mechanics."""
    points = _capture_points(local_target_points, geometry=geometry, opening_m=opening_m)
    x, y, z = points.T
    tip = geometry.palm_z_m + geometry.finger_length_m
    inside = ((np.abs(x) <= opening_m / 2)
              & (np.abs(y) <= geometry.pad_height_m / 2)
              & (z >= geometry.palm_z_m) & (z <= tip))
    count = int(np.count_nonzero(inside))
    score = 0.0 if not count else float(np.median(np.clip(
        (tip - z[inside]) / (0.2 * geometry.finger_length_m), 0.0, 1.0)))
    return count, score


@dataclass(frozen=True)
class ParallelJawPadHeightSupport:
    """Relative support of a local pad-height patch, not whole-object coverage."""

    score: float
    center_shift_m: float
    weighted_points: float
    best_weighted_points: float


@dataclass(frozen=True)
class ParallelJawOpposedCaptureSupport:
    """Observed insertion on both closing edges, not antipodal contact proof."""

    score: float
    side_scores: tuple[float, float]
    side_point_counts: tuple[int, int]


def estimate_grasp_surface_normals(
    target_points: np.ndarray, *, radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate request-local PCA normals without modifying collision points.

    Distinct neighbours, a metric radius and covariance rank keep replicated
    pixels, isolated samples and lines from manufacturing a surface. The
    returned confidence is planarity, not probability of a successful grasp.
    Normal signs are deliberately unspecified; closure compatibility is axial.
    """
    points = np.asarray(target_points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Surface normals require finite Nx3 points.")
    if isinstance(radius_m, bool) or not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Surface normal radius must be finite and positive.")
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    normals = np.zeros_like(unique)
    confidence = np.zeros(len(unique))
    if len(unique) >= 6:
        distance, indices = cKDTree(unique).query(unique, k=min(12, len(unique)))
        inside = distance <= radius_m
        count = inside.sum(axis=1)
        neighbours = unique[indices]
        mean = (neighbours * inside[..., None]).sum(axis=1) / count[:, None]
        centered = (neighbours - mean[:, None]) * inside[..., None]
        covariance = np.einsum("nki,nkj->nij", centered, centered) / count[:, None, None]
        values, vectors = np.linalg.eigh(covariance)
        # Scale-relative rank: neither a line nor a near-isotropic patch
        # establishes a stable normal. Sparse/ambiguous patches stay unknown.
        valid = ((count >= 6) & (values[:, 1] - values[:, 0] > values[:, 2] * .1)
                 & (values[:, 2] > 0))
        normals[valid] = vectors[valid, :, 0]
        confidence[valid] = np.clip(
            (values[valid, 1] - values[valid, 0]) / values[valid, 2], 0., 1.)
    return normals[inverse], confidence[inverse]


@dataclass(frozen=True)
class ParallelJawSurfaceAlignment:
    """Observed closure evidence, not antipodal or physical contact proof.

    ``observed_sides`` counts spatial bands with reliable normals. Only the
    continuous opposition term distinguishes separated surfaces from two
    bands on one face; two observed bands alone never establish two contacts.
    """

    score: float
    observed_sides: int
    side_point_counts: tuple[int, int]
    opposition_score: float = 0.0
    pad_height_score: float = 0.0
    center_shift_m: float = 0.0
    height_evidence_weight: float = 0.0


def _weighted_surface_quartile(values: np.ndarray, weights: np.ndarray) -> float:
    """Empirical upper quartile; zero-weight samples convey no observation."""
    positive = weights > 0.
    if not np.any(positive):
        return 0.
    values, weights = values[positive], weights[positive]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, .75*cumulative[-1])), len(order)-1)
    return float(values[order[index]])


def parallel_jaw_surface_alignment(
    local_target_points: np.ndarray, local_normals: np.ndarray,
    normal_confidence: np.ndarray, *, geometry: ParallelJawSolidGeometry,
    opening_m: float,
) -> ParallelJawSurfaceAlignment:
    """Prefer closure toward a locally observed surface, independent of gravity.

    A single face earns at most half credit. The second band earns extra
    credit only to the extent that the vector between band centres has a
    normal component on both surfaces. Coplanar bands cannot manufacture an
    opposed pair by spanning X. All projections are sign-invariant; PCA normal
    signs need not be oriented or guessed from an unseen surface.

    Normal-weighted local pad support favours the observed closing faces over
    a dense top face. Its centre is only an advisory translation proposal. No
    world-up, known object size, centroid containment or new rejection is used.
    """
    points = _capture_points(local_target_points, geometry=geometry, opening_m=opening_m)
    normals = np.asarray(local_normals, dtype=float)
    confidence = np.asarray(normal_confidence, dtype=float)
    lengths = np.linalg.norm(normals, axis=1) if normals.shape == points.shape else np.array([])
    if (normals.shape != points.shape or confidence.shape != (len(points),)
            or not np.isfinite(normals).all() or not np.isfinite(confidence).all()
            or np.any((confidence < 0) | (confidence > 1))
            or np.any((confidence > 0) & ~np.isclose(lengths, 1., atol=1e-7, rtol=0))):
        raise ValueError("Surface alignment requires matching unit normals and bounded confidence.")
    # Rigid candidate transforms preserve distinctness, so deduplicate once,
    # not independently for the band kernel and each closing edge.
    _, distinct = np.unique(points, axis=0, return_index=True)
    points, normals, confidence = points[distinct], normals[distinct], confidence[distinct]
    directional_evidence = np.clip(confidence * normals[:, 0]**2, 0., 1.)
    x, y, z = points.T
    band = ((np.abs(x) <= opening_m / 2) & (np.abs(y) <= geometry.pad_height_m / 2)
            & (z >= geometry.palm_z_m) & (z <= geometry.palm_z_m + geometry.finger_length_m)
            & (confidence > 0))
    # Distinct observations keep replicated pixels from changing either the
    # contact band or its centre. Zero directional weight conveys no contact
    # height information and cannot move an otherwise supported band.
    # Compare neighbouring bands around this pinch, not the densest remote
    # part of a bottle/tool. Keep one pad-height of context on either side of
    # the actual half-height window; do not infer a whole-object centre.
    neighbourhood = np.abs(y) <= 1.5*geometry.pad_height_m
    contact_height = parallel_jaw_pad_height_support(
        points[neighbourhood], geometry=geometry, opening_m=opening_m,
        point_weights=directional_evidence[neighbourhood])
    if not np.any(band):
        return ParallelJawSurfaceAlignment(0., 0, (0, 0), 0.,
            contact_height.score, contact_height.center_shift_m,
            contact_height.best_weighted_points)
    low, high = float(np.min(x[band])), float(np.max(x[band]))
    quarter = (high - low) / 4
    # An X-normal face may have zero X span: it is one observed face, not no
    # evidence and not two coincident opposing faces.
    edges = ((np.ones(len(points), dtype=bool),) if quarter <= geometry.pad_height_m*1e-9
             else (x <= low + quarter, x >= high - quarter))
    scores, counts, patches = [], [], []
    for edge in edges:
        selected = band & edge & (confidence > 0)
        counts.append(int(np.count_nonzero(selected)))
        if counts[-1] >= 3:
            evidence = directional_evidence[selected]
            # Unrelated normals cannot qualify one or two directional
            # outliers as a contact patch. Cap both the vote and centre weight
            # at the third-largest distinct directional observation.
            supported = np.partition(evidence, len(evidence)-3)[-3]
            weights = np.minimum(evidence, supported)
            scores.append(_weighted_surface_quartile(weights, weights))
            patches.append((points[selected], normals[selected], weights))
    opposition = 0.
    if len(patches) == 2:
        centres = [np.average(p, axis=0, weights=w) if np.any(w > 0.) else np.mean(p, axis=0)
                   for p, _n, w in patches]
        separation = centres[1] - centres[0]
        distance = float(np.linalg.norm(separation))
        if distance > geometry.pad_height_m*1e-9:
            direction = separation / distance
            opposition = min(_weighted_surface_quartile(
                np.clip((n @ direction)**2, 0., 1.), w) for _p, n, w in patches)
            resolved = min(1., distance*math.sqrt(opposition) /
                (_SURFACE_PAIR_RESOLUTION_PAD_FRACTION*geometry.pad_height_m))
            opposition *= resolved
    scores += [0.] * (2-len(scores))
    # Removing a weak band cannot increase the one-face credit or the paired
    # bonus. Unobserved support stays unknown, never a synthetic opposing face.
    score = (.5*max(scores) + .5*min(scores)*opposition) * contact_height.score
    return ParallelJawSurfaceAlignment(
        float(np.clip(score, 0., 1.)), len(patches), tuple(counts + [0]*(2-len(counts))),
        opposition, contact_height.score, contact_height.center_shift_m,
        contact_height.best_weighted_points)


def parallel_jaw_opposed_capture_support(
    local_target_points: np.ndarray, *, geometry: ParallelJawSolidGeometry,
    opening_m: float,
) -> ParallelJawOpposedCaptureSupport:
    """Do not let deep central points conceal one fingertip missing the target.

    Inspect the outer quarters of the local band's spatial X span separately.
    Point-count quantiles would let dense central samples move the boundaries
    inward and conceal an observed miss at the outer closing edge.
    An outer band earns positive support only with a patch of three distinct
    points within a pad-half-height neighbourhood. A single extreme point
    cannot move the boundary and manufacture an opposed supporting surface.
    Keep a 20%-finger-length band beyond the tips as zero-insertion evidence;
    cropping at the tip first would censor the very points that reveal a miss.
    Ignore farther object parts, preserving a local pinch on long objects.
    Each side's upper score quartile tolerates sparse edge noise and partial
    visibility. Missing support lowers a soft preference, never proves an
    impossible grasp or authorizes rejection of unseen geometry.
    """
    points = _capture_points(local_target_points, geometry=geometry, opening_m=opening_m)
    x, y, z = points.T
    depth = .2 * geometry.finger_length_m
    tip = geometry.palm_z_m + geometry.finger_length_m
    band = ((np.abs(x) <= opening_m/2) & (np.abs(y) <= geometry.pad_height_m/2)
            & (z >= geometry.palm_z_m) & (z <= tip+depth))
    x, z = x[band], z[band]
    if not len(x):
        return ParallelJawOpposedCaptureSupport(0., (0., 0.), (0, 0))
    low, high = float(np.min(x)), float(np.max(x))
    # A single observed line/point is not two distinct closing surfaces.
    if high-low <= 1e-9:
        return ParallelJawOpposedCaptureSupport(0., (0., 0.), (0, 0))
    quarter = (high-low)/4
    local = points[band]
    sides = tuple(np.unique(local[side], axis=0) for side in (x <= low+quarter, x >= high-quarter))
    # Empirical-CDF quantiles are unchanged by replicating every observation;
    # interpolating order statistics depend on sample count at sparse edges.
    def side_score(side):
        if len(side) < 3:
            return 0.
        neighbours = cKDTree(side).query(side, k=3)[0][:, -1]
        if not np.any(neighbours <= geometry.pad_height_m/2):
            return 0.
        # Keep every observed zero-insertion point in this band's score;
        # neighbourhood support is not permission to delete negative evidence.
        return float(np.quantile(np.clip((tip-side[:, 2])/depth, 0., 1.), .75,
                                 method="inverted_cdf"))
    scores = tuple(side_score(side) for side in sides)
    return ParallelJawOpposedCaptureSupport(min(scores), scores, tuple(len(side) for side in sides))


def parallel_jaw_pad_height_support(
    local_target_points: np.ndarray, *, geometry: ParallelJawSolidGeometry,
    opening_m: float, point_weights: np.ndarray | None = None,
) -> ParallelJawPadHeightSupport:
    """Compare this pad band with the best translated band on observed points.

    A triangular kernel vanishes at either pad edge. Its peak convolution is
    attained at an observed Y value; sorted prefix sums evaluate every such
    centre without a grid or pairwise distance matrix. Normalizing by the best
    local band preserves tall-object pinches and rejects a grazing top rim
    even when axial insertion is excellent. Optional bounded observation
    weights separate closing-face support from unrelated surface density;
    omitting them preserves the existing unweighted admission metric.
    No unseen volume is inferred.
    """
    points = _capture_points(local_target_points, geometry=geometry, opening_m=opening_m)
    weights = np.ones(len(points)) if point_weights is None else np.asarray(point_weights, dtype=float)
    if (weights.shape != (len(points),) or not np.isfinite(weights).all()
            or np.any((weights < 0.) | (weights > 1.))):
        raise ValueError("Pad support weights must match points and be finite in [0, 1].")
    x, y, z = points.T
    selected = ((np.abs(x) <= opening_m/2) & (z >= geometry.palm_z_m)
                & (z <= geometry.palm_z_m+geometry.finger_length_m) & (weights > 0.))
    order = np.argsort(y[selected], kind="stable")
    ys, ws = y[selected][order], weights[selected][order]
    if not len(ys):
        return ParallelJawPadHeightSupport(0., 0., 0., 0.)
    half = geometry.pad_height_m/2
    prefix = np.concatenate(([0.], np.cumsum(ys*ws)))
    prefix_weights = np.concatenate(([0.], np.cumsum(ws)))
    left = np.searchsorted(ys, ys-half, side="left")
    middle = np.searchsorted(ys, ys, side="right")
    right = np.searchsorted(ys, ys+half, side="right")
    distance_sums = (ys*(prefix_weights[middle]-prefix_weights[left])-(prefix[middle]-prefix[left])
                     + prefix[right]-prefix[middle]-ys*(prefix_weights[right]-prefix_weights[middle]))
    band_weights = np.clip(prefix_weights[right]-prefix_weights[left]-distance_sums/half, 0., np.sum(ws))
    best = float(np.max(band_weights))
    tied = np.flatnonzero(np.isclose(band_weights, best, rtol=1e-12, atol=1e-9*np.sum(ws)))
    selected = min(tied, key=lambda i: (abs(ys[i]), ys[i]))
    current = float(np.sum(ws*np.clip(1-np.abs(ys)/half, 0., 1.)))
    return ParallelJawPadHeightSupport(
        float(np.clip(current/best, 0., 1.)), float(ys[selected]), current, best)


class VisibleGraspScene:
    def __init__(self, *, points: np.ndarray, target_mask: np.ndarray,
                 geometry: ParallelJawSolidGeometry,
                 capture_qpos_by_side: Mapping[str, np.ndarray] | None = None,
                 capture_self_observation_evidence: (
                     Mapping[str, object] | None
                 ) = None,
                 depth_preprocessing_evidence: (
                     Mapping[str, object] | None
                 ) = None) -> None:
        xyz = np.asarray(points, dtype=float)
        labels = np.asarray(target_mask)
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("scene must contain finite Nx3 XYZ points")
        if labels.dtype != np.bool_ or labels.shape != (len(xyz),):
            raise ValueError("scene target labels must be a boolean N vector")
        self.points = xyz.copy()
        self.target_mask = labels.copy()
        self.geometry = geometry
        self.capture_qpos_by_side = None
        if capture_qpos_by_side is not None:
            if set(capture_qpos_by_side) != {"left", "right"}:
                raise ValueError("capture qpos must retain both arms")
            qpos = {side: np.asarray(q, dtype=float).copy()
                    for side, q in capture_qpos_by_side.items()}
            if any(q.shape != (6,) or not np.all(np.isfinite(q)) for q in qpos.values()):
                raise ValueError("capture qpos must contain six finite radians per arm")
            for q in qpos.values():
                q.setflags(write=False)
            self.capture_qpos_by_side = MappingProxyType(qpos)
        self.capture_self_observation_evidence = None
        if capture_self_observation_evidence is not None:
            evidence = dict(capture_self_observation_evidence)
            removed_background = evidence.get("removed_background_points")
            if (
                isinstance(removed_background, bool)
                or not isinstance(removed_background, int)
                or removed_background < 0
                or evidence.get("removed_target_points") != 0
            ):
                raise ValueError(
                    "capture self-observation evidence must retain targets and "
                    "contain a nonnegative background count"
                )
            canonical_content_hash(
                evidence,
                domain="dexteleop.motion.capture_self_observation_evidence.v1",
            )
            self.capture_self_observation_evidence = MappingProxyType(evidence)
        self.depth_preprocessing_evidence = None
        if depth_preprocessing_evidence is not None:
            evidence = dict(depth_preprocessing_evidence)
            removed_background = evidence.get(
                "removed_isolated_background_points"
            )
            if (
                evidence.get("schema")
                != "dexteleop.motion.visible_grasp_depth_preprocessing.v1"
                or isinstance(removed_background, bool)
                or not isinstance(removed_background, int)
                or removed_background < 0
                or evidence.get("removed_target_points") != 0
            ):
                raise ValueError(
                    "depth preprocessing evidence must preserve target points "
                    "and contain a nonnegative background count"
                )
            canonical_content_hash(
                evidence,
                domain="dexteleop.motion.visible_grasp_depth_preprocessing.v1",
            )
            self.depth_preprocessing_evidence = MappingProxyType(evidence)
        self.points.setflags(write=False)
        self.target_mask.setflags(write=False)
        self._tree = cKDTree(self.points)
        self.semantic_content_hash = canonical_content_hash({
            "points_sha256": hashlib.sha256(self.points.tobytes()).hexdigest(),
            "target_labels_sha256": hashlib.sha256(self.target_mask.tobytes()).hexdigest(),
            "geometry": asdict(geometry),
            "capture_self_observation_evidence": (
                None
                if self.capture_self_observation_evidence is None
                else dict(self.capture_self_observation_evidence)
            ),
            "depth_preprocessing_evidence": (
                None
                if self.depth_preprocessing_evidence is None
                else dict(self.depth_preprocessing_evidence)
            ),
        }, domain="dexteleop.motion.visible_grasp_scene.v1")

    def _local_points(self, pose: np.ndarray, *, opening_m: float,
                      distance_m: float = 0, proxy_padding_m: float = 0) -> tuple[np.ndarray, np.ndarray]:
        g = self.geometry
        low_z = g.palm_z_m - g.finger_thickness_m - distance_m - proxy_padding_m
        high_z = g.palm_z_m + g.finger_length_m + proxy_padding_m
        center_z = (low_z + high_z) / 2
        # Enclose the same solids about their own center, not the TCP. The
        # inverse also preserves the local-point predicate for accepted poses
        # that are only numerically orthonormal.
        inverse = np.linalg.inv(pose[:3, :3])
        center = pose[:3, 3] + center_z * inverse[2]
        extents = np.array((opening_m/2 + g.finger_thickness_m + proxy_padding_m,
                            g.pad_height_m/2 + proxy_padding_m, (high_z - low_z)/2))
        corners = np.array(((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)))
        radius = float(np.linalg.norm((corners * extents) @ inverse, axis=1).max())
        # Reuse one native index array for XYZ and labels. Passing the Python
        # list to both NumPy lookups repeats its conversion for every pose.
        indices = np.asarray(self._tree.query_ball_point(center, radius+1e-9),
                             dtype=np.intp)
        return ((self.points[indices]-pose[:3, 3]) @ pose[:3, :3],
                self.target_mask[indices])

    @classmethod
    def from_depth(cls, *, depth_m: np.ndarray, target_mask: np.ndarray,
                   intrinsics: Mapping[str, float], reference_from_camera: np.ndarray,
                   geometry: ParallelJawSolidGeometry,
                   capture_qpos_by_side: Mapping[str, np.ndarray] | None = None,
                   capture_self_observation_evidence: (
                       Mapping[str, object] | None
                   ) = None,
                   depth_filter: VisibleGraspDepthFilterConfig | None = None,
                   ) -> VisibleGraspScene:
        depth = np.asarray(depth_m)
        mask = np.asarray(target_mask)
        if (depth.ndim != 2 or mask.shape != depth.shape or mask.dtype != np.bool_
                or not np.all(np.isfinite(depth)) or np.any(depth < 0)):
            raise ValueError("depth and boolean target mask must be aligned")
        fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
        if not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
            raise ValueError("invalid intrinsics")
        transform = np.asarray(reference_from_camera, dtype=float)
        if (transform.shape != (4, 4) or not np.all(np.isfinite(transform))
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1)):
            raise ValueError("camera transform must be rigid")
        depth_preprocessing_evidence = None
        valid_depth = depth > 0
        if depth_filter is not None:
            if not isinstance(depth_filter, VisibleGraspDepthFilterConfig):
                raise TypeError("depth_filter must be VisibleGraspDepthFilterConfig or None")
            valid_depth, depth_preprocessing_evidence = (
                filter_isolated_background_depth(
                    depth_m=depth,
                    target_mask=mask,
                    config=depth_filter,
                )
            )
        rows, columns = np.nonzero(valid_depth)
        z = depth[rows, columns]
        xyz = np.column_stack(((columns-cx)*z/fx, (rows-cy)*z/fy, z))
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        return cls(points=xyz, target_mask=mask[rows, columns], geometry=geometry,
                   capture_qpos_by_side=capture_qpos_by_side,
                   capture_self_observation_evidence=(
                       capture_self_observation_evidence
                   ),
                   depth_preprocessing_evidence=depth_preprocessing_evidence)

    def _solid_mask(self, local: np.ndarray, *, opening_m: float,
                    distance_m: float = 0, proxy_padding_m: float = 0) -> np.ndarray:
        """Use the same separate palm/finger solids for filtering and checks."""
        g = self.geometry
        x, y, z = local.T
        fingers = self._finger_mask(local, opening_m=opening_m,
                                    distance_m=distance_m, proxy_padding_m=proxy_padding_m)
        palm = ((z > g.palm_z_m-g.finger_thickness_m-distance_m-proxy_padding_m)
                & (z <= g.palm_z_m+proxy_padding_m)
                & (np.abs(x) < opening_m/2+g.finger_thickness_m+proxy_padding_m))
        return (np.abs(y) < g.pad_height_m/2+proxy_padding_m) & (fingers | palm)

    def _finger_mask(self, local: np.ndarray, *, opening_m: float,
                     distance_m: float = 0, proxy_padding_m: float = 0) -> np.ndarray:
        g = self.geometry
        x, _y, z = local.T
        return (
            (z > g.palm_z_m - distance_m - proxy_padding_m)
            & (z < g.palm_z_m + g.finger_length_m + proxy_padding_m)
            & (np.abs(x) > opening_m / 2 - proxy_padding_m)
            & (np.abs(x) < opening_m / 2 + g.finger_thickness_m + proxy_padding_m)
        )

    def _terminal_target_contact_mask(
        self,
        local: np.ndarray,
        *,
        opening_m: float,
        contact_depth_m: float,
    ) -> np.ndarray:
        """Return target points allowed to enter the inner tactile-pad skin.

        The parallel jaws close along local X.  Expected grasp contact is
        therefore on each finger's inner X face, not in a local-Z band at the
        distal fingertip.  ``contact_depth_m`` is a bounded penetration allowance
        into that inner face for depth/segmentation uncertainty; it never opens
        the palm, the outer finger faces, or any background collision pair.
        """
        if (
            not math.isfinite(contact_depth_m)
            or contact_depth_m <= 0.0
            or contact_depth_m > self.geometry.finger_thickness_m
        ):
            raise ValueError(
                "contact_depth_m must be finite and within finger thickness"
            )
        x, y, z = local.T
        inner_face_x = opening_m / 2.0
        return (
            (np.abs(x) > inner_face_x)
            & (np.abs(x) <= inner_face_x + contact_depth_m)
            & (z > self.geometry.palm_z_m)
            & (z < self.geometry.palm_z_m + self.geometry.finger_length_m)
            & (np.abs(y) < self.geometry.pad_height_m / 2)
        )

    def without_captured_gripper_observations(
        self,
        tool_tcp_poses: Mapping[str, np.ndarray],
        *,
        surface_margin_m: float = 0.0,
    ) -> tuple[VisibleGraspScene, int]:
        """Remove background attributable to open gripper solids at capture time.

        This is robot self-observation filtering, not collision-pair permission.
        Capture opening is approximated by maximum_opening_m, as in the path
        checks. A bounded surface margin accounts for depth samples on the
        observed solid boundary; it never fills the finger gap, extends an
        approach sweep, or removes target-labelled points. The resulting scene
        stays fixed while the robot moves in the planner.
        """
        margin = float(surface_margin_m)
        if (
            isinstance(surface_margin_m, bool)
            or not math.isfinite(margin)
            or not 0.0 <= margin <= self.geometry.finger_thickness_m
        ):
            raise ValueError(
                "surface_margin_m must be finite and within finger thickness"
            )
        occupied = np.zeros(len(self.points), dtype=bool)
        g = self.geometry
        for pose in tool_tcp_poses.values():
            self.collision_counts(pose, opening_m=g.maximum_opening_m)
            pose = g.solid_pose(np.asarray(pose, dtype=float))
            local = (self.points-pose[:3, 3]) @ pose[:3, :3]
            if margin == 0.0:
                occupied |= self._solid_mask(
                    local, opening_m=g.maximum_opening_m
                )
                continue
            x, y, z = local.T
            fingers = (
                (z >= g.palm_z_m - margin)
                & (z <= g.palm_z_m + g.finger_length_m + margin)
                & (np.abs(x) >= max(0.0, g.maximum_opening_m / 2.0 - margin))
                & (
                    np.abs(x)
                    <= g.maximum_opening_m / 2.0 + g.finger_thickness_m + margin
                )
            )
            palm = (
                (z >= g.palm_z_m - g.finger_thickness_m - margin)
                & (z <= g.palm_z_m + margin)
                & (
                    np.abs(x)
                    <= g.maximum_opening_m / 2.0 + g.finger_thickness_m + margin
                )
            )
            occupied |= (np.abs(y) <= g.pad_height_m / 2.0 + margin) & (
                fingers | palm
            )
        remove = occupied & ~self.target_mask
        return (VisibleGraspScene(points=self.points[~remove],
                                  target_mask=self.target_mask[~remove],
                                  geometry=g,
                                  capture_qpos_by_side=self.capture_qpos_by_side,
                                  capture_self_observation_evidence=(
                                      self.capture_self_observation_evidence
                                  ),
                                  depth_preprocessing_evidence=(
                                      self.depth_preprocessing_evidence
                                  )),
                int(np.count_nonzero(remove)))

    def _validated_solid_pose(
        self, pose: np.ndarray, *, opening_m: float,
    ) -> np.ndarray:
        g = self.geometry
        if not math.isfinite(opening_m) or not 0 < opening_m <= g.maximum_opening_m:
            raise ValueError("opening is outside the geometry bound")
        pose = np.asarray(pose, dtype=float)
        identity = np.eye(3)
        # Preserve the previous allclose/isclose tolerances for finite poses;
        # these small fixed-shape comparisons are hot in candidate and path checks.
        if (pose.shape != (4, 4) or not np.all(np.isfinite(pose))
                or not np.all(np.abs(pose[3] - (0, 0, 0, 1))
                              <= (1e-8, 1e-8, 1e-8, 1e-8 + 1e-5))
                or not np.all(np.abs(pose[:3, :3].T @ pose[:3, :3] - identity)
                              <= 1e-7 + 1e-5 * identity)
                or not abs(float(np.linalg.det(pose[:3, :3])) - 1.) <= 1e-8 + 1e-5):
            raise ValueError("grasp pose must be rigid")
        return g.solid_pose(pose)


    def _collision_local_points(
        self, pose: np.ndarray, *, opening_m: float, proxy_padding_m: float = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (isinstance(proxy_padding_m, bool) or not math.isfinite(proxy_padding_m)
                or proxy_padding_m < 0):
            raise ValueError("proxy_padding_m must be finite and nonnegative")
        pose = self._validated_solid_pose(pose, opening_m=opening_m)
        return self._local_points(pose, opening_m=opening_m, proxy_padding_m=proxy_padding_m)
    def collision_counts(self, pose: np.ndarray, *, opening_m: float,
                         proxy_padding_m: float = 0,
                         target_proxy_padding_m: float | None = None) -> tuple[int, int]:
        """Count points in separate box solids, optionally padded by the path owner.

        Padding expands each face, not the finger gap's entire bounding box.
        The default is nominal: perception, endpoint checks and self filtering
        keep their existing geometry. Target padding can taper at grasp contact
        without reducing the background padding.
        """
        target_padding = proxy_padding_m if target_proxy_padding_m is None else target_proxy_padding_m
        if (isinstance(target_padding, bool) or not math.isfinite(target_padding)
                or not 0 <= target_padding <= proxy_padding_m):
            raise ValueError("target_proxy_padding_m must be within proxy padding")
        local, labels = self._collision_local_points(pose, opening_m=opening_m,
                                                     proxy_padding_m=proxy_padding_m)
        occupied = self._solid_mask(local, opening_m=opening_m, proxy_padding_m=proxy_padding_m)
        target = occupied if target_padding == proxy_padding_m else self._solid_mask(
            local, opening_m=opening_m, proxy_padding_m=target_padding)
        return (int(np.count_nonzero(target & labels)),
                int(np.count_nonzero(occupied & ~labels)))

    def collision_counts_with_terminal_target_contact(
        self,
        pose: np.ndarray,
        *,
        opening_m: float,
        contact_depth_m: float,
        proxy_padding_m: float = 0,
    ) -> tuple[int, int, int]:
        """Separate allowed inner-pad contact from target and background hits.

        The target is still treated as an obstacle everywhere except the
        terminal inner tactile-pad skin.  This is the grasp-goal contact rule; callers
        that need a strict no-contact check should continue using
        :meth:`collision_counts`.
        """
        # Planned grasp contact keeps nominal target geometry and pad rules;
        # padding must not turn the intentional finger/target contact into a
        # forbidden clearance request. Background solids stay padded.
        local, labels = self._collision_local_points(pose, opening_m=opening_m,
                                                     proxy_padding_m=proxy_padding_m)
        occupied = self._solid_mask(local, opening_m=opening_m)
        target = occupied & labels
        terminal_contact = self._terminal_target_contact_mask(
            local, opening_m=opening_m, contact_depth_m=contact_depth_m
        )
        background_occupied = occupied if proxy_padding_m == 0 else self._solid_mask(
            local, opening_m=opening_m, proxy_padding_m=proxy_padding_m)
        return (
            int(np.count_nonzero(target & ~terminal_contact)),
            int(np.count_nonzero(target & terminal_contact)),
            int(np.count_nonzero(background_occupied & ~labels)),
        )

    def approach_collision_counts(
        self,
        grasp_pose: np.ndarray,
        *,
        opening_m: float,
        distance_m: float,
        allow_terminal_target_contact: bool = False,
        terminal_target_contact_depth_m: float = 0.006,
    ) -> tuple[int, int]:
        """Check the entire fixed-orientation +Z approach analytically.

        In the final TCP frame, a stationary point moves only along local Z;
        expanding each solid backwards by the approach distance is therefore
        its exact swept volume. No waypoint sampling can skip a thin obstacle.
        """
        if not math.isfinite(distance_m) or distance_m < 0:
            raise ValueError("approach distance must be finite and nonnegative")
        # Reuse the same input checks, not an unused terminal collision query.
        pose = self._validated_solid_pose(grasp_pose, opening_m=opening_m)
        local, labels = self._local_points(pose, opening_m=opening_m, distance_m=distance_m)
        occupied = self._solid_mask(local, opening_m=opening_m, distance_m=distance_m)
        target = occupied & labels
        if allow_terminal_target_contact:
            target &= ~self._terminal_target_contact_mask(
                local,
                opening_m=opening_m,
                contact_depth_m=terminal_target_contact_depth_m,
            )
        return (int(np.count_nonzero(target)),
                int(np.count_nonzero(occupied & ~labels)))

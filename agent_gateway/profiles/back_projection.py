"""Shared live-depth back-projection registration, geometry, and executor."""

from __future__ import annotations

import math
import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from robot_motion.object_transport import (
    CalibratedBoxSupportObservation,
    SupportSurfaceObservation,
)

from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
)
from agent_gateway.protocol.skill_io_contracts import SkillIOContract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.back_projection_schemas import (
    back_project_schema_entry_v1,
    validate_back_project_parameters_v1,
    validate_back_project_result_for_parameters_v1,
    validate_back_project_result_v1,
)
from agent_gateway.skills.perception.definition import (
    back_project_skill_definition_v1,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash

BACK_PROJECT_MIN_REGION_VALID_POINTS_V1 = 8
BACK_PROJECT_MAX_REGION_PIXELS_V1 = 1_048_576
BACK_PROJECT_MAX_DEPTH_M_V1 = 10.0


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _digest(value: object, *, field_name: str) -> str:
    text = _required(value, field_name=field_name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return text


def _matrix(value: object, *, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} must be a finite 4x4 transform.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{field_name} has an invalid homogeneous row.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{field_name} rotation is not orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{field_name} rotation is not right-handed.")
    return matrix.copy()


def _source_catalog(
    values: tuple[Mapping[str, object], ...],
) -> tuple[tuple[str, str, int, int], ...]:
    expected = {"source_id", "role", "native_width", "native_height"}
    normalized = []
    for item in values:
        if not isinstance(item, Mapping) or set(item) != expected:
            raise ValueError("Back-projection source entry has unexpected fields.")
        source_id = _required(item["source_id"], field_name="source_id")
        role = _required(item["role"], field_name="role")
        width = item["native_width"]
        height = item["native_height"]
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise ValueError("Back-projection dimensions must be positive integers.")
        normalized.append((source_id, role, width, height))
    normalized.sort(key=lambda item: item[1])
    if (
        not normalized
        or len({item[0] for item in normalized}) != len(normalized)
        or len({item[1] for item in normalized}) != len(normalized)
    ):
        raise ValueError("Back-projection sources and roles must be unique.")
    return tuple(normalized)


@dataclass(frozen=True)
class LiveAlignedDepthCapture:
    """One immutable metric-depth cut and its Station-frame camera pose."""

    source_id: str
    role: str
    captured_at: float
    station_revision: int
    capture_group_id: str
    depth_content_hash: str
    intrinsics: Mapping[str, float | int]
    reference_from_camera: np.ndarray
    depth_m: np.ndarray

    def __post_init__(self) -> None:
        _required(self.source_id, field_name="source_id")
        _required(self.role, field_name="role")
        _required(self.capture_group_id, field_name="capture_group_id")
        _digest(self.depth_content_hash, field_name="depth_content_hash")
        if not math.isfinite(float(self.captured_at)):
            raise ValueError("captured_at must be finite.")
        if (
            isinstance(self.station_revision, bool)
            or not isinstance(self.station_revision, int)
            or self.station_revision < 0
        ):
            raise ValueError("station_revision must be non-negative.")
        depth = np.asarray(self.depth_m, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError("depth_m must be one non-empty 2D array.")
        if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
            raise ValueError("depth_m must contain finite non-negative meters.")
        intrinsics = dict(self.intrinsics)
        required = {"width", "height", "fx", "fy", "cx", "cy"}
        if not required.issubset(intrinsics):
            raise ValueError("intrinsics omit required pinhole fields.")
        width = intrinsics["width"]
        height = intrinsics["height"]
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or depth.shape != (height, width)
        ):
            raise ValueError("intrinsics dimensions differ from depth_m.")
        for name in ("fx", "fy", "cx", "cy"):
            value = intrinsics[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"intrinsics.{name} must be finite.")
        if float(intrinsics["fx"]) <= 0.0 or float(intrinsics["fy"]) <= 0.0:
            raise ValueError("intrinsics focal lengths must be positive.")
        transform = _matrix(
            self.reference_from_camera,
            field_name="reference_from_camera",
        )
        depth = np.ascontiguousarray(depth)
        depth.setflags(write=False)
        transform.setflags(write=False)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "intrinsics", MappingProxyType(intrinsics))
        object.__setattr__(self, "reference_from_camera", transform)


class LiveAlignedDepthCapturePort(Protocol):
    def capture_aligned_depth(
        self, *, role: str, expected_station_revision: int
    ) -> LiveAlignedDepthCapture: ...


@dataclass(frozen=True)
class CorrelatedPlacementCapture:
    """One RGB image and aligned depth cut captured under one backend owner."""

    rgb: LiveRgbSegmentationCapture
    depth: LiveAlignedDepthCapture
    rgb_jpeg: bytes | None = None
    depth_npy: bytes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rgb, LiveRgbSegmentationCapture) or not isinstance(
            self.depth, LiveAlignedDepthCapture
        ):
            raise TypeError("Placement RGB-D capture members must be typed.")
        if (
            self.rgb.source_id != self.depth.source_id
            or self.rgb.role != self.depth.role
            or self.rgb.station_revision != self.depth.station_revision
            or self.rgb.image.size != self.depth.depth_m.shape[::-1]
        ):
            raise ValueError("Placement RGB and depth cuts do not share one identity.")


class CorrelatedPlacementCapturePort(Protocol):
    def capture_placement_rgbd(
        self, *, role: str, expected_station_revision: int
    ) -> CorrelatedPlacementCapture: ...


@dataclass(frozen=True)
class ProjectedDepthGeometry:
    world_center_xyz_m: tuple[float, float, float]
    world_median_xyz_m: tuple[float, float, float]
    world_bounds_low_xyz_m: tuple[float, float, float]
    world_bounds_high_xyz_m: tuple[float, float, float]
    world_extent_xyz_m: tuple[float, float, float]
    median_depth_m: float
    valid_point_count: int


@dataclass(frozen=True)
class ProjectedSupportSurface:
    """Top-plane candidate, not proof that the entire XY AABB is supported."""

    source_id: str
    capture_group_id: str
    captured_at: float
    station_revision: int
    depth_content_hash: str
    upper_top_z_m: float
    plane_anchor_reference_xyz_m: tuple[float, float, float]
    normal_reference_xyz: tuple[float, float, float]
    observed_bounds_min_xy_m: tuple[float, float]
    observed_bounds_max_xy_m: tuple[float, float]
    maximum_plane_residual_m: float
    top_point_count: int


@dataclass(frozen=True)
class SearchedSupportSurface:
    """Best measured planar patch; its bounds are still only observations."""

    candidate: ProjectedSupportSurface
    patch_xyxy_px: tuple[int, int, int, int]
    inspected_patch_count: int
    accepted_patch_count: int
    support_mask_sha256: str
    source_mask_clipped: bool


def bind_known_box_support_from_depth(
    capture: LiveAlignedDepthCapture,
    search: SearchedSupportSurface,
    box: CalibratedBoxSupportObservation,
    *,
    semantic_support_id: str,
    expected_reference_frame_id: str,
    expected_support_mask_sha256: str,
    lateral_uncertainty_m: float,
    vertical_uncertainty_m: float,
) -> SupportSurfaceObservation:
    """Bind a retained semantic mask and same-cut plane to known box geometry.

    The caller must obtain the semantic ID and expected mask hash from the
    authenticated segmentation operation. This edge never grants motion;
    the lower safe planner and live observer still validate the result.
    """
    if (not isinstance(capture, LiveAlignedDepthCapture)
            or not isinstance(search, SearchedSupportSurface)
            or not isinstance(box, CalibratedBoxSupportObservation)):
        raise TypeError("Support fusion needs typed depth, search and box observations")
    candidate = search.candidate
    if (candidate.source_id != capture.source_id
            or candidate.capture_group_id != capture.capture_group_id
            or candidate.captured_at != capture.captured_at
            or candidate.station_revision != capture.station_revision
            or candidate.depth_content_hash != capture.depth_content_hash):
        raise ValueError("Support plane does not belong to the bound depth cut")
    if (semantic_support_id != box.support_id
            or box.reference_frame_id != expected_reference_frame_id
            or box.scene_revision != capture.station_revision):
        raise ValueError("Semantic support, frame or revision differs from its box binding")
    if (not isinstance(expected_support_mask_sha256, str)
            or len(expected_support_mask_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_support_mask_sha256)
            or search.support_mask_sha256 != expected_support_mask_sha256):
        raise ValueError("Support mask differs from the retained semantic segment")
    center_xy = (np.asarray(candidate.observed_bounds_min_xy_m)
                 + candidate.observed_bounds_max_xy_m) / 2.
    depth_ref = ("support-depth:" + capture.depth_content_hash + ":"
                 + search.support_mask_sha256 + ":" + capture.capture_group_id)
    return box.bind_measured_top(
        plane_measured_at=capture.captured_at,
        measured_upper_top_z_m=candidate.upper_top_z_m,
        measured_normal_reference_xyz=candidate.normal_reference_xyz,
        measured_region_center_xy_m=tuple(float(value) for value in center_xy),
        depth_evidence_ref=depth_ref,
        lateral_uncertainty_m=lateral_uncertainty_m,
        vertical_uncertainty_m=vertical_uncertainty_m,
    )


def search_and_bind_known_box_support_from_depth(
    capture: LiveAlignedDepthCapture,
    mask: np.ndarray,
    box: CalibratedBoxSupportObservation,
    *,
    mask_clipped: bool,
    semantic_support_id: str,
    expected_reference_frame_id: str,
    expected_support_mask_sha256: str,
    lateral_uncertainty_m: float,
    vertical_uncertainty_m: float,
    maximum_depth_m: float,
    patch_size_px: int = 64,
    stride_px: int = 32,
    maximum_patches: int = 512,
) -> tuple[SearchedSupportSurface, SupportSurfaceObservation]:
    """Find a measured patch that actually matches the named complete box.

    A broad semantic mask can include nearby same-colour surfaces or borders.
    Candidate ranking is restricted to patches whose measured pose and plane
    agree with the catalog-backed box. The retained semantic mask hash still
    names the *original* mask, not a privately cropped replacement.
    """
    selected = np.asarray(mask)
    if (not isinstance(capture, LiveAlignedDepthCapture)
            or not isinstance(box, CalibratedBoxSupportObservation)
            or selected.dtype != np.bool_ or selected.shape != capture.depth_m.shape):
        raise TypeError("Bound support search needs typed depth, box and mask")
    actual_mask_hash = hashlib.sha256(
        np.ascontiguousarray(selected).tobytes()
    ).hexdigest()
    if (not isinstance(expected_support_mask_sha256, str)
            or actual_mask_hash != expected_support_mask_sha256):
        raise ValueError("Support search mask differs from its retained segment")
    if (semantic_support_id != box.support_id
            or box.reference_frame_id != expected_reference_frame_id
            or box.scene_revision != capture.station_revision):
        raise ValueError("Support search identity, frame or revision differs")
    depth_ref = ("support-depth:" + capture.depth_content_hash + ":"
                 + actual_mask_hash + ":" + capture.capture_group_id)

    def matches_box(candidate: ProjectedSupportSurface) -> bool:
        center_xy = (np.asarray(candidate.observed_bounds_min_xy_m)
                     + candidate.observed_bounds_max_xy_m) / 2.
        try:
            box.bind_measured_top(
                plane_measured_at=capture.captured_at,
                measured_upper_top_z_m=candidate.upper_top_z_m,
                measured_normal_reference_xyz=candidate.normal_reference_xyz,
                measured_region_center_xy_m=tuple(float(value) for value in center_xy),
                depth_evidence_ref=depth_ref,
                lateral_uncertainty_m=lateral_uncertainty_m,
                vertical_uncertainty_m=vertical_uncertainty_m,
            )
        except ValueError:
            return False
        return True

    search = search_depth_mask_support_surface(
        capture, selected, mask_clipped=mask_clipped,
        maximum_depth_m=maximum_depth_m, patch_size_px=patch_size_px,
        stride_px=stride_px, maximum_patches=maximum_patches,
        candidate_filter=matches_box,
    )
    support = bind_known_box_support_from_depth(
        capture, search, box, semantic_support_id=semantic_support_id,
        expected_reference_frame_id=expected_reference_frame_id,
        expected_support_mask_sha256=expected_support_mask_sha256,
        lateral_uncertainty_m=lateral_uncertainty_m,
        vertical_uncertainty_m=vertical_uncertainty_m,
    )
    return search, support


def refresh_known_box_support_patch_from_depth(
    source_capture: LiveAlignedDepthCapture,
    fresh_capture: LiveAlignedDepthCapture,
    semantic_mask: np.ndarray,
    selection: SearchedSupportSurface,
    fresh_box: CalibratedBoxSupportObservation,
    *,
    semantic_support_id: str,
    expected_reference_frame_id: str,
    expected_support_mask_sha256: str,
    lateral_uncertainty_m: float,
    vertical_uncertainty_m: float,
    maximum_depth_m: float,
) -> tuple[SearchedSupportSurface, SupportSurfaceObservation]:
    """Revalidate only a selected patch on a fresh cut before motion planning.

    The expensive semantic search may run on an earlier cut. This method
    preserves its exact mask and static camera identity but never refreshes a
    stale plane by copying a timestamp. It fits the selected patch again on
    actual fresh depth, then binds a freshly read fixed support box.
    """
    if (not isinstance(source_capture, LiveAlignedDepthCapture)
            or not isinstance(fresh_capture, LiveAlignedDepthCapture)
            or not isinstance(selection, SearchedSupportSurface)):
        raise TypeError("Support refresh needs typed source, fresh cut and selection")
    source = selection.candidate
    if (source.source_id != source_capture.source_id
            or source.capture_group_id != source_capture.capture_group_id
            or source.depth_content_hash != source_capture.depth_content_hash
            or source.captured_at != source_capture.captured_at
            or source.station_revision != source_capture.station_revision):
        raise ValueError("Selected patch does not belong to its source cut")
    if (source_capture.source_id != fresh_capture.source_id
            or source_capture.role != fresh_capture.role
            or source_capture.station_revision != fresh_capture.station_revision
            or dict(source_capture.intrinsics) != dict(fresh_capture.intrinsics)
            or not np.allclose(source_capture.reference_from_camera,
                               fresh_capture.reference_from_camera,
                               atol=1e-6, rtol=0)):
        raise ValueError("Saved semantic mask cannot be reused after camera binding changed")
    selected = np.asarray(semantic_mask)
    if selected.dtype != np.bool_ or selected.shape != source_capture.depth_m.shape:
        raise ValueError("Retained semantic mask has the wrong dimensions")
    mask_hash = hashlib.sha256(np.ascontiguousarray(selected).tobytes()).hexdigest()
    if (not isinstance(expected_support_mask_sha256, str)
            or mask_hash != expected_support_mask_sha256
            or mask_hash != selection.support_mask_sha256):
        raise ValueError("Retained support mask identity changed")
    patch = selection.patch_xyxy_px
    width, height = selected.shape[1], selected.shape[0]
    if (len(patch) != 4 or any(isinstance(value, bool) or not isinstance(value, int)
                               for value in patch)):
        raise ValueError("Selected support patch coordinates are invalid")
    x0, y0, x1, y1 = patch
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Selected support patch exceeds the camera grid")
    local_mask = np.zeros_like(selected)
    local_mask[y0:y1, x0:x1] = selected[y0:y1, x0:x1]
    fresh_candidate = project_depth_mask_support_surface(
        fresh_capture, local_mask, mask_clipped=False,
        maximum_depth_m=maximum_depth_m)
    refreshed = SearchedSupportSurface(
        candidate=fresh_candidate, patch_xyxy_px=patch,
        inspected_patch_count=1, accepted_patch_count=1,
        support_mask_sha256=mask_hash,
        source_mask_clipped=selection.source_mask_clipped,
    )
    support = bind_known_box_support_from_depth(
        fresh_capture, refreshed, fresh_box,
        semantic_support_id=semantic_support_id,
        expected_reference_frame_id=expected_reference_frame_id,
        expected_support_mask_sha256=mask_hash,
        lateral_uncertainty_m=lateral_uncertainty_m,
        vertical_uncertainty_m=vertical_uncertainty_m,
    )
    return refreshed, support


@dataclass(frozen=True)
class ObservedSupportFootprintCoverage:
    """Dense RGB-D evidence for one proposed rectangle, not complete CAD geometry."""

    region_min_xy_m: tuple[float, float]
    region_max_xy_m: tuple[float, float]
    sample_count: int
    maximum_measured_plane_residual_m: float
    resolution_m: float


def check_observed_support_footprint(
    capture: LiveAlignedDepthCapture,
    mask: np.ndarray,
    candidate: ProjectedSupportSurface,
    *,
    region_min_xy_m: tuple[float, float],
    region_max_xy_m: tuple[float, float],
    resolution_m: float = .002,
    maximum_plane_residual_m: float = .002,
) -> ObservedSupportFootprintCoverage:
    """Require measured planar depth at every sampled point of a proposed footprint.

    This closes gaps in an AABB of observed pixels at 2 mm resolution or finer.
    It cannot prove subpixel material continuity or hidden support geometry;
    callers must still bind a known support extent and fresh identity before
    constructing a complete SupportSurfaceObservation.
    """
    if not isinstance(capture, LiveAlignedDepthCapture) or not isinstance(candidate, ProjectedSupportSurface):
        raise TypeError("Support footprint needs a bound capture and fitted plane")
    if (candidate.source_id != capture.source_id
            or candidate.capture_group_id != capture.capture_group_id
            or candidate.captured_at != capture.captured_at
            or candidate.station_revision != capture.station_revision
            or candidate.depth_content_hash != capture.depth_content_hash):
        raise ValueError("Support plane and footprint depth have different source cuts")
    selected = np.asarray(mask)
    if selected.dtype != np.bool_ or selected.shape != capture.depth_m.shape:
        raise ValueError("Support mask must match the bound depth cut")
    for name, value in (("resolution_m", resolution_m),
                        ("maximum_plane_residual_m", maximum_plane_residual_m)):
        if (isinstance(value, (bool, np.bool_)) or not math.isfinite(value)
                or value <= 0 or value > .002):
            raise ValueError(f"{name} exceeds the measured support qualification profile")
    low = np.asarray(region_min_xy_m, dtype=float)
    high = np.asarray(region_max_xy_m, dtype=float)
    observed_low = np.asarray(candidate.observed_bounds_min_xy_m)
    observed_high = np.asarray(candidate.observed_bounds_max_xy_m)
    if (low.shape != (2,) or high.shape != (2,) or not np.isfinite(low).all()
            or not np.isfinite(high).all() or np.any(low >= high)
            or np.any(low < observed_low) or np.any(high > observed_high)):
        raise ValueError("Requested footprint is outside the observed support bounds")
    steps = (high - low) / resolution_m
    if (not np.isfinite(steps).all() or np.any(steps > 100_000)
            or (steps[0] + 1) * (steps[1] + 1) > 100_000):
        raise ValueError("Requested footprint exceeds the bounded sample budget")
    counts = (math.ceil(float(steps[0])) + 1,
              math.ceil(float(steps[1])) + 1)
    if counts[0] * counts[1] > 100_000:
        raise ValueError("Requested footprint exceeds the bounded sample budget")
    grid_x, grid_y = np.meshgrid(np.linspace(low[0], high[0], counts[0]),
                                 np.linspace(low[1], high[1], counts[1]))
    anchor = np.asarray(candidate.plane_anchor_reference_xyz_m)
    normal = np.asarray(candidate.normal_reference_xyz)
    x = grid_x.ravel()
    y = grid_y.ravel()
    z = anchor[2] - (normal[0] * (x - anchor[0])
                     + normal[1] * (y - anchor[1])) / normal[2]
    reference_points = np.stack((x, y, z), axis=1)
    rotation = capture.reference_from_camera[:3, :3]
    camera_points = (reference_points - capture.reference_from_camera[:3, 3]) @ rotation
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("Support footprint projects behind the camera")
    intrinsics = capture.intrinsics
    u = np.rint(intrinsics["fx"] * camera_points[:, 0] / camera_points[:, 2]
                + intrinsics["cx"]).astype(int)
    v = np.rint(intrinsics["fy"] * camera_points[:, 1] / camera_points[:, 2]
                + intrinsics["cy"]).astype(int)
    height, width = selected.shape
    if np.any(u < 0) or np.any(v < 0) or np.any(u >= width) or np.any(v >= height):
        raise ValueError("Support footprint is outside the captured image")
    if not np.all(selected[v, u]):
        raise ValueError("Support footprint contains an unobserved mask gap")
    measured_depth = capture.depth_m[v, u].astype(float)
    if np.any(measured_depth <= 0):
        raise ValueError("Support footprint contains missing metric depth")
    measured = _project_pixels(
        u=u.astype(float), v=v.astype(float), depth=measured_depth,
        intrinsics=intrinsics,
        reference_from_camera=capture.reference_from_camera,
    )
    residual = np.abs((measured - anchor) @ normal)
    if np.any(residual > maximum_plane_residual_m):
        raise ValueError("Support footprint contains nonplanar depth or an occlusion")
    return ObservedSupportFootprintCoverage(
        region_min_xy_m=tuple(float(value) for value in low),
        region_max_xy_m=tuple(float(value) for value in high),
        sample_count=int(residual.size),
        maximum_measured_plane_residual_m=float(np.max(residual)),
        resolution_m=float(resolution_m),
    )


def search_depth_mask_support_surface(
    capture: LiveAlignedDepthCapture,
    mask: np.ndarray,
    *,
    mask_clipped: bool,
    maximum_depth_m: float,
    patch_size_px: int = 64,
    stride_px: int = 32,
    maximum_patches: int = 256,
    candidate_filter: Callable[[ProjectedSupportSurface], bool] | None = None,
) -> SearchedSupportSurface:
    """Select a measured top patch inside one bound semantic support mask.

    A large support mask can include edges and occlusions, so fitting it all at
    once may reject a valid local plane. This bounded search only finds a
    candidate. It does not certify a complete footprint or motion eligibility.
    The semantic mask identity, clipping status and depth cut remain the
    caller's responsibility.
    """
    if not isinstance(capture, LiveAlignedDepthCapture):
        raise TypeError("A bound metric-depth capture is required")
    selected = np.asarray(mask)
    if selected.dtype != np.bool_ or selected.shape != capture.depth_m.shape:
        raise ValueError("Support mask must be a paired full-resolution boolean grid")
    if type(mask_clipped) is not bool:
        raise ValueError("Support mask clipping status must be explicit")
    if (isinstance(maximum_depth_m, (bool, np.bool_))
            or not isinstance(maximum_depth_m, (int, float, np.integer, np.floating))
            or not math.isfinite(maximum_depth_m) or maximum_depth_m <= 0):
        raise ValueError("maximum_depth_m must be finite and positive")
    for name, value in (
        ("patch_size_px", patch_size_px),
        ("stride_px", stride_px),
        ("maximum_patches", maximum_patches),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if patch_size_px < 16 or stride_px > patch_size_px or maximum_patches > 1024:
        raise ValueError("Support patch search bounds are invalid")
    if candidate_filter is not None and not callable(candidate_filter):
        raise TypeError("Support patch filter must be callable")
    height, width = selected.shape
    if patch_size_px > min(height, width):
        raise ValueError("Support patch exceeds the capture")
    rows, columns = np.nonzero(selected)
    if not len(rows):
        raise ValueError("Support mask is empty")

    def starts(low: int, high: int, limit: int) -> tuple[int, ...]:
        first = max(0, low - patch_size_px + 1)
        last = min(high, limit - patch_size_px)
        return tuple(range(first, last + 1, stride_px)) + ((last,) if last >= first else ())

    xs = tuple(dict.fromkeys(starts(int(columns.min()), int(columns.max()), width)))
    ys = tuple(dict.fromkeys(starts(int(rows.min()), int(rows.max()), height)))
    if len(xs) * len(ys) > maximum_patches:
        raise ValueError("Support mask exceeds the bounded patch search budget")
    best: tuple[float, float, int, int, ProjectedSupportSurface] | None = None
    accepted = 0
    for y0 in ys:
        for x0 in xs:
            patch = np.zeros_like(selected)
            patch[y0:y0 + patch_size_px, x0:x0 + patch_size_px] = selected[
                y0:y0 + patch_size_px, x0:x0 + patch_size_px
            ]
            try:
                candidate = project_depth_mask_support_surface(
                    capture, patch, mask_clipped=False,
                    maximum_depth_m=maximum_depth_m,
                )
            except ValueError:
                continue
            if candidate_filter is not None and not candidate_filter(candidate):
                continue
            accepted += 1
            dx = (candidate.observed_bounds_max_xy_m[0]
                  - candidate.observed_bounds_min_xy_m[0])
            dy = (candidate.observed_bounds_max_xy_m[1]
                  - candidate.observed_bounds_min_xy_m[1])
            score = (dx * dy, -candidate.maximum_plane_residual_m,
                     -y0, -x0, candidate)
            if best is None or score[:4] > best[:4]:
                best = score
    if best is None:
        raise ValueError("No near-horizontal measured support patch was found")
    x0, y0 = -best[3], -best[2]
    return SearchedSupportSurface(
        candidate=best[4],
        patch_xyxy_px=(x0, y0, x0 + patch_size_px, y0 + patch_size_px),
        inspected_patch_count=len(xs) * len(ys),
        accepted_patch_count=accepted,
        support_mask_sha256=hashlib.sha256(
            np.ascontiguousarray(selected).tobytes()
        ).hexdigest(),
        source_mask_clipped=mask_clipped,
    )


def project_depth_mask_support_surface(
    capture: LiveAlignedDepthCapture,
    mask: np.ndarray,
    *,
    mask_clipped: bool,
    maximum_depth_m: float,
    minimum_top_points: int = 32,
    maximum_tilt_deg: float = 1.,
    maximum_plane_residual_m: float = .002,
    top_height_margin_m: float = .002,
) -> ProjectedSupportSurface:
    """Fit only a measured top patch; reject sides, clipping and poor geometry.

    The returned XY bounds are only quantiles of observed top points. They are
    not automatically an inscribed supported rectangle: rotated surfaces,
    holes and occlusion need separate conservative qualification before use as
    SupportSurfaceObservation. upper_top_z_m includes a positive error margin.
    This operation is read-only and does not assert identity or motion safety.
    """
    if not isinstance(capture, LiveAlignedDepthCapture):
        raise TypeError("A bound metric-depth capture is required")
    selected = np.asarray(mask)
    if selected.dtype != np.bool_ or selected.shape != capture.depth_m.shape:
        raise ValueError("Support mask must be a paired full-resolution boolean grid")
    if type(mask_clipped) is not bool or mask_clipped:
        raise ValueError("Clipped or unverified support masks are not complete regions")
    if (isinstance(minimum_top_points, bool) or not isinstance(minimum_top_points, int)
            or minimum_top_points < 16):
        raise ValueError("Support plane needs at least sixteen measured top points")
    for name, value in (("maximum_depth_m", maximum_depth_m),
                        ("maximum_tilt_deg", maximum_tilt_deg),
                        ("maximum_plane_residual_m", maximum_plane_residual_m),
                        ("top_height_margin_m", top_height_margin_m)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if maximum_tilt_deg > 1. or maximum_plane_residual_m > .002 or top_height_margin_m > .005:
        raise ValueError("Support plane and uncertainty limits exceed the release geometry profile")
    valid = (selected & np.isfinite(capture.depth_m)
             & (capture.depth_m > 0.) & (capture.depth_m <= maximum_depth_m))
    rows, cols = np.nonzero(valid)
    if len(rows) < minimum_top_points:
        raise ValueError("Support mask has insufficient valid metric depth")
    points = _project_pixels(
        u=cols.astype(np.float64), v=rows.astype(np.float64),
        depth=capture.depth_m[valid].astype(np.float64),
        intrinsics=capture.intrinsics,
        reference_from_camera=capture.reference_from_camera)
    threshold = float(np.quantile(points[:, 2], .8))
    top = points[points[:, 2] >= threshold - 1e-9]
    if len(top) < minimum_top_points:
        raise ValueError("Support top has insufficient measured coverage")
    centered = top - np.mean(top, axis=0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    if singular[1] <= 1e-5:
        raise ValueError("Support top is geometrically degenerate")
    normal = vectors[-1]
    if normal[2] < 0:
        normal = -normal
    if normal[2] < math.cos(math.radians(maximum_tilt_deg)):
        raise ValueError("Observed support top is not near-horizontal")
    residual = np.abs(centered @ normal)
    max_residual = float(np.quantile(residual, .95))
    if max_residual > maximum_plane_residual_m:
        raise ValueError("Observed support top does not fit a stable plane")
    xy_bounds = np.quantile(top[:, :2], [.05, .95], axis=0)
    if np.any(xy_bounds[1] <= xy_bounds[0]):
        raise ValueError("Observed support top has no finite placement region")
    return ProjectedSupportSurface(
        source_id=capture.source_id,
        capture_group_id=capture.capture_group_id,
        captured_at=capture.captured_at,
        station_revision=capture.station_revision,
        depth_content_hash=capture.depth_content_hash,
        upper_top_z_m=float(np.max(top[:, 2]) + top_height_margin_m),
        plane_anchor_reference_xyz_m=tuple(float(value) for value in np.mean(top, axis=0)),
        normal_reference_xyz=tuple(float(value) for value in normal),
        observed_bounds_min_xy_m=tuple(float(value) for value in xy_bounds[0]),
        observed_bounds_max_xy_m=tuple(float(value) for value in xy_bounds[1]),
        maximum_plane_residual_m=max_residual, top_point_count=len(top))


# Inclined supports (an inclined board) have their own measurement entry: the
# near-horizontal path above keeps its 1 degree invariant and its "highest 20 %"
# top band, which on a slope is only a narrow strip at the upper end.
MAXIMUM_INCLINED_SUPPORT_TILT_DEG = 30.


def project_depth_mask_inclined_support_surface(
    capture: LiveAlignedDepthCapture,
    mask: np.ndarray,
    *,
    mask_clipped: bool,
    maximum_depth_m: float,
    maximum_tilt_deg: float = MAXIMUM_INCLINED_SUPPORT_TILT_DEG,
    minimum_top_points: int = 32,
    maximum_plane_residual_m: float = .002,
    top_height_margin_m: float = .002,
    minimum_inlier_fraction: float = .6,
) -> ProjectedSupportSurface:
    """Fit the dominant plane of a support mask that may be inclined (<= 30 degrees).

    All valid mask points are fitted (trimmed least squares: three refits on
    the 70 % of points nearest the current plane), then the points within the
    plane residual are the measured top; they must be at least
    ``minimum_inlier_fraction`` of the mask, so a side face or an object
    resting on the support cannot carry the fit. Same result type and
    residual bound as the near-horizontal path; read-only.
    """
    if not isinstance(capture, LiveAlignedDepthCapture):
        raise TypeError("A bound metric-depth capture is required")
    selected = np.asarray(mask)
    if selected.dtype != np.bool_ or selected.shape != capture.depth_m.shape:
        raise ValueError("Support mask must be a paired full-resolution boolean grid")
    if type(mask_clipped) is not bool or mask_clipped:
        raise ValueError("Clipped or unverified support masks are not complete regions")
    if (isinstance(minimum_top_points, bool) or not isinstance(minimum_top_points, int)
            or minimum_top_points < 16):
        raise ValueError("Support plane needs at least sixteen measured top points")
    for name, value in (("maximum_depth_m", maximum_depth_m),
                        ("maximum_tilt_deg", maximum_tilt_deg),
                        ("maximum_plane_residual_m", maximum_plane_residual_m),
                        ("top_height_margin_m", top_height_margin_m),
                        ("minimum_inlier_fraction", minimum_inlier_fraction)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (maximum_tilt_deg > MAXIMUM_INCLINED_SUPPORT_TILT_DEG or maximum_plane_residual_m > .002
            or top_height_margin_m > .005 or minimum_inlier_fraction > 1.):
        raise ValueError("Support plane and uncertainty limits exceed the release geometry profile")
    valid = (selected & np.isfinite(capture.depth_m)
             & (capture.depth_m > 0.) & (capture.depth_m <= maximum_depth_m))
    rows, cols = np.nonzero(valid)
    if len(rows) < minimum_top_points:
        raise ValueError("Support mask has insufficient valid metric depth")
    points = _project_pixels(
        u=cols.astype(np.float64), v=rows.astype(np.float64),
        depth=capture.depth_m[valid].astype(np.float64),
        intrinsics=capture.intrinsics,
        reference_from_camera=capture.reference_from_camera)

    def fit(subset):
        center = np.mean(subset, axis=0)
        _, singular, vectors = np.linalg.svd(subset - center, full_matrices=False)
        if singular[1] <= 1e-5:
            raise ValueError("Support top is geometrically degenerate")
        normal = vectors[-1]
        return center, (normal if normal[2] >= 0 else -normal)

    center, normal = fit(points)
    for _ in range(3):
        distance = np.abs((points - center) @ normal)
        keep = distance <= np.quantile(distance, .7)
        if int(np.count_nonzero(keep)) < minimum_top_points:
            break
        center, normal = fit(points[keep])
    distance = np.abs((points - center) @ normal)
    top = points[distance <= maximum_plane_residual_m]
    if len(top) < minimum_top_points or len(top) < minimum_inlier_fraction * len(points):
        raise ValueError("Observed support top does not fit a stable plane")
    center, normal = fit(top)
    if normal[2] < math.cos(math.radians(maximum_tilt_deg)):
        raise ValueError("Observed support top is steeper than the inclined support profile")
    residual = np.abs((top - center) @ normal)
    max_residual = float(np.quantile(residual, .95))
    if max_residual > maximum_plane_residual_m:
        raise ValueError("Observed support top does not fit a stable plane")
    xy_bounds = np.quantile(top[:, :2], [.05, .95], axis=0)
    if np.any(xy_bounds[1] <= xy_bounds[0]):
        raise ValueError("Observed support top has no finite placement region")
    return ProjectedSupportSurface(
        source_id=capture.source_id,
        capture_group_id=capture.capture_group_id,
        captured_at=capture.captured_at,
        station_revision=capture.station_revision,
        depth_content_hash=capture.depth_content_hash,
        upper_top_z_m=float(np.max(top[:, 2]) + top_height_margin_m),
        plane_anchor_reference_xyz_m=tuple(float(value) for value in center),
        normal_reference_xyz=tuple(float(value) for value in normal),
        observed_bounds_min_xy_m=tuple(float(value) for value in xy_bounds[0]),
        observed_bounds_max_xy_m=tuple(float(value) for value in xy_bounds[1]),
        maximum_plane_residual_m=max_residual, top_point_count=len(top))


def project_depth_mask_geometry(
    capture: LiveAlignedDepthCapture,
    mask: np.ndarray,
    *,
    minimum_valid_points: int,
    maximum_depth_m: float,
    robust_lower_quantile: float = 0.05,
    robust_upper_quantile: float = 0.95,
    world_z_min_m: float | None = None,
    world_z_max_m: float | None = None,
    insufficient_points_error: str = "depth_mask_insufficient_points",
) -> ProjectedDepthGeometry:
    """Project exact selected mask pixels into the capture's reference frame."""

    selected = np.asarray(mask, dtype=bool)
    if selected.shape != capture.depth_m.shape:
        raise ValueError("Depth mask dimensions differ from the live depth image.")
    if (
        isinstance(minimum_valid_points, bool)
        or not isinstance(minimum_valid_points, int)
        or minimum_valid_points <= 0
    ):
        raise ValueError("minimum_valid_points must be positive.")
    if not math.isfinite(maximum_depth_m) or maximum_depth_m <= 0.0:
        raise ValueError("maximum_depth_m must be positive and finite.")
    if not 0.0 <= robust_lower_quantile < robust_upper_quantile <= 1.0:
        raise ValueError("Robust quantiles are invalid.")
    for name, value in (
        ("world_z_min_m", world_z_min_m),
        ("world_z_max_m", world_z_max_m),
    ):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite when provided.")
    if (
        world_z_min_m is not None
        and world_z_max_m is not None
        and world_z_min_m > world_z_max_m
    ):
        raise ValueError("world_z_min_m must not exceed world_z_max_m.")
    valid = (
        selected
        & np.isfinite(capture.depth_m)
        & (capture.depth_m > 0.0)
        & (capture.depth_m <= maximum_depth_m)
    )
    rows, columns = np.nonzero(valid)
    depths = capture.depth_m[valid].astype(np.float64, copy=False)
    points = _project_pixels(
        u=columns.astype(np.float64),
        v=rows.astype(np.float64),
        depth=depths,
        intrinsics=capture.intrinsics,
        reference_from_camera=capture.reference_from_camera,
    )
    keep = np.ones(points.shape[0], dtype=bool)
    if world_z_min_m is not None:
        keep &= points[:, 2] >= float(world_z_min_m)
    if world_z_max_m is not None:
        keep &= points[:, 2] <= float(world_z_max_m)
    points = points[keep]
    depths = depths[keep]
    if depths.size < minimum_valid_points:
        raise RuntimeError(insufficient_points_error)
    low, high = np.quantile(
        points,
        [robust_lower_quantile, robust_upper_quantile],
        axis=0,
    )
    median = np.median(points, axis=0)
    center = (low + high) / 2.0
    center[2] = median[2]
    extent = high - low

    def vector(values: np.ndarray) -> tuple[float, float, float]:
        return (
            round(float(values[0]), 6),
            round(float(values[1]), 6),
            round(float(values[2]), 6),
        )

    return ProjectedDepthGeometry(
        world_center_xyz_m=vector(center),
        world_median_xyz_m=vector(median),
        world_bounds_low_xyz_m=vector(low),
        world_bounds_high_xyz_m=vector(high),
        world_extent_xyz_m=vector(extent),
        median_depth_m=round(float(np.median(depths)), 6),
        valid_point_count=int(depths.size),
    )


@dataclass(frozen=True)
class BackProjectionProfile:
    target_kind: str
    current_mode: str
    sources: tuple[tuple[str, str, int, int], ...]
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    minimum_region_valid_points: int = BACK_PROJECT_MIN_REGION_VALID_POINTS_V1
    maximum_region_pixels: int = BACK_PROJECT_MAX_REGION_PIXELS_V1
    maximum_depth_m: float = BACK_PROJECT_MAX_DEPTH_M_V1
    robust_lower_quantile: float = 0.05
    robust_upper_quantile: float = 0.95
    profile_id: str = "live-aligned-depth-back-projection-v1"

    def __post_init__(self) -> None:
        if self.target_kind not in {"real", "sim"}:
            raise ValueError("Back-projection target_kind is unsupported.")
        if self.current_mode not in {"real_observe", "real_active", "sim_active"}:
            raise ValueError("Back-projection current_mode is unsupported.")
        if (self.target_kind == "real") != self.current_mode.startswith("real_"):
            raise ValueError("Back-projection target kind and mode disagree.")
        for field_name in (
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "profile_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        sources = _source_catalog(
            tuple(
                {
                    "source_id": source_id,
                    "role": role,
                    "native_width": width,
                    "native_height": height,
                }
                for source_id, role, width, height in self.sources
            )
        )
        object.__setattr__(self, "sources", sources)
        for field_name in ("minimum_region_valid_points", "maximum_region_pixels"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if not math.isfinite(self.maximum_depth_m) or self.maximum_depth_m <= 0.0:
            raise ValueError("maximum_depth_m must be positive and finite.")
        if not (
            0.0 <= self.robust_lower_quantile
            < self.robust_upper_quantile
            <= 1.0
        ):
            raise ValueError("Back-projection robust quantiles are invalid.")

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(item[1] for item in self.sources)

    @property
    def source_id_by_role(self) -> Mapping[str, str]:
        return {role: source for source, role, _width, _height in self.sources}

    @property
    def dimensions_by_role(self) -> Mapping[str, tuple[int, int]]:
        return {
            role: (width, height)
            for _source, role, width, height in self.sources
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "target_kind": self.target_kind,
                "current_mode": self.current_mode,
                "sources": self.sources,
                "reference_frame_id": self.reference_frame_id,
                "reference_frame_spec_hash": self.reference_frame_spec_hash,
                "frame_graph_hash": self.frame_graph_hash,
                "minimum_region_valid_points": self.minimum_region_valid_points,
                "maximum_region_pixels": self.maximum_region_pixels,
                "maximum_depth_m": self.maximum_depth_m,
                "robust_quantiles": [
                    self.robust_lower_quantile,
                    self.robust_upper_quantile,
                ],
                "pixel_convention": "uv_0based",
                "region_convention": "xyxy_0based_half_open",
                "downstream_control_effects": "forbidden",
            },
            domain="dexteleop.agent_skill.back_projection_profile.v1",
        )


@dataclass(frozen=True)
class BackProjectionDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    target_kind: str
    executor_id: str
    deployment_binding_version: str = "1"

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                field_name: getattr(self, field_name)
                for field_name in self.__dataclass_fields__
            }
            | {
                "control_effect_class": "read_only",
                "downstream_control_effects": "forbidden",
            },
            domain="dexteleop.agent_skill.back_projection_deployment.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind=self.target_kind,
            execution_manifest_hash=self.execution_manifest_hash,
        )


@dataclass(frozen=True)
class BackProjectionRegistration:
    profile: BackProjectionProfile
    skill: SkillRegistrationTemplate
    binding: BackProjectionDeploymentBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile, BackProjectionProfile):
            raise TypeError("profile must be a BackProjectionProfile.")
        if not isinstance(self.skill, SkillRegistrationTemplate):
            raise TypeError("skill must be a SkillRegistrationTemplate.")
        if not isinstance(self.binding, BackProjectionDeploymentBinding):
            raise TypeError("binding must be a BackProjectionDeploymentBinding.")
        if not isinstance(self.payload_validator, AgentSkillPayloadValidatorSet):
            raise TypeError("payload_validator must be typed.")
        if len(self.skill.routes) != 1:
            raise ValueError("Back-projection registration requires exactly one Route.")
        route = self.skill.routes[0]
        definition = back_project_skill_definition_v1()
        if self.skill.definition != definition or (
            route.skill_id,
            route.skill_version,
            route.control_effect_class,
            route.privilege_tier,
            route.may_move_robot,
            route.may_open_publication,
            route.requires_operator_confirmation,
            route.requires_deadman,
            route.max_operation_duration_ms,
            route.cancellation_policy,
            route.requires_task_evaluation,
        ) != (
            definition.skill_id,
            definition.skill_version,
            "read_only",
            "t0_read_only",
            False,
            False,
            False,
            False,
            3_000,
            "cancel_only",
            False,
        ):
            raise ValueError("Back-projection registration uses another contract.")
        profile_refs = route.allowed_parameter_profiles
        if len(profile_refs) != 1 or (
            profile_refs[0].parameter_profile_id != self.profile.profile_id
            or profile_refs[0].parameter_profile_hash
            != self.profile.semantic_content_hash
        ):
            raise ValueError("Back-projection Route does not bind its exact profile.")
        if (
            route.target_kind != self.profile.target_kind
            or route.supported_modes != (self.profile.current_mode,)
            or self.binding.route_hash != route.semantic_content_hash
            or self.binding.profile_hash != self.profile.semantic_content_hash
            or self.binding.target_kind != self.profile.target_kind
        ):
            raise ValueError("Back-projection deployment contracts disagree.")
        self.payload_validator.validate_route(route)
        expected_availability_hash = canonical_content_hash(
            {
                "status": "configured_requires_live_aligned_depth",
                "route_hash": route.semantic_content_hash,
                "deployment_binding_hash": self.binding.semantic_content_hash,
                "profile_hash": self.profile.semantic_content_hash,
            },
            domain="dexteleop.agent_skill.back_projection_availability.v1",
        )
        if self.availability_hash != expected_availability_hash:
            raise ValueError("Back-projection availability changed its contracts.")


def build_back_projection_registration(
    *,
    target_kind: str,
    current_mode: str,
    source_catalog: tuple[Mapping[str, object], ...],
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
) -> BackProjectionRegistration:
    profile = BackProjectionProfile(
        target_kind=target_kind,
        current_mode=current_mode,
        sources=_source_catalog(source_catalog),
        reference_frame_id=reference_frame_id,
        reference_frame_spec_hash=reference_frame_spec_hash,
        frame_graph_hash=frame_graph_hash,
    )
    definition = back_project_skill_definition_v1()
    schema = back_project_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_roles=profile.roles,
        reference_frame_id=profile.reference_frame_id,
        reference_frame_spec_hash=profile.reference_frame_spec_hash,
        frame_graph_hash=profile.frame_graph_hash,
    )
    io_contract = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=f"{target_kind}.live_depth.perception.back_project",
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=(current_mode,),
        target_kind=target_kind,
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=3_000,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
        skill_io_contract_hash=io_contract.semantic_content_hash,
    )
    skill = SkillRegistrationTemplate(
        definition=definition,
        schema_entry=schema,
        io_contract=io_contract,
        routes=(route,),
    )
    binding = BackProjectionDeploymentBinding(
        deployment_binding_id=f"{target_kind}.live_depth.back_projection:{runtime_session_id}",
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
        target_kind=target_kind,
        executor_id=f"{target_kind}-passive-depth-back-projection.v1",
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id=f"dexteleop.agent_skill.validator.{target_kind}_back_projection:{runtime_session_id}",
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda payload: validate_back_project_parameters_v1(
            payload, allowed_roles=profile.roles
        ),
        validate_execution_result=lambda payload: validate_back_project_result_v1(
            payload,
            source_id_by_role=profile.source_id_by_role,
            reference_frame_id=profile.reference_frame_id,
            reference_frame_spec_hash=profile.reference_frame_spec_hash,
            frame_graph_hash=profile.frame_graph_hash,
            minimum_region_valid_points=profile.minimum_region_valid_points,
            maximum_depth_m=profile.maximum_depth_m,
        ),
        validate_execution_result_for_parameters=(
            validate_back_project_result_for_parameters_v1
        ),
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    availability_hash = canonical_content_hash(
        {
            "status": "configured_requires_live_aligned_depth",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.back_projection_availability.v1",
    )
    return BackProjectionRegistration(
        profile=profile,
        skill=skill,
        binding=binding,
        payload_validator=validator,
        availability_hash=availability_hash,
    )


class LiveBackProjectionProvider:
    def __init__(
        self,
        *,
        capture_port: LiveAlignedDepthCapturePort,
        registration: BackProjectionRegistration,
    ) -> None:
        if not callable(getattr(capture_port, "capture_aligned_depth", None)):
            raise TypeError("capture_port must expose capture_aligned_depth().")
        if not isinstance(registration, BackProjectionRegistration):
            raise TypeError("registration must be typed.")
        self.capture_port = capture_port
        self.registration = registration

    def project_request(self, request: Any) -> dict[str, Any]:
        result, _capture = self.project_parameters(
            request.parameters,
            expected_station_revision=request.expected_station_revision,
        )
        return result


    def project_parameters(
        self, parameters: Mapping[str, Any], *, expected_station_revision: int
    ) -> tuple[dict[str, Any], LiveAlignedDepthCapture]:
        """Return geometry and its exact capture without a second depth read."""

        role = str(parameters["role"])
        capture = self.capture_port.capture_aligned_depth(
            role=role,
            expected_station_revision=expected_station_revision,
        )
        return self.project_capture(
            capture, parameters, expected_station_revision=expected_station_revision
        ), capture


    def project_capture(
        self, capture: LiveAlignedDepthCapture, parameters: Mapping[str, Any],
        *, expected_station_revision: int,
    ) -> dict[str, Any]:
        """Project a previously captured cut without reading the source again."""

        self.registration.payload_validator.validate_parameters(parameters)
        profile = self.registration.profile
        role = str(parameters["role"])
        if (
            capture.role != role
            or capture.source_id != profile.source_id_by_role[role]
            or capture.station_revision != expected_station_revision
            or capture.depth_m.shape[::-1] != profile.dimensions_by_role[role]
        ):
            raise RuntimeError("Back-projection capture changed its bound identity.")
        pixel = parameters["target_pixel_uv"]
        region = parameters["pixel_region_xyxy"]
        selected = np.zeros(capture.depth_m.shape, dtype=bool)
        if pixel is not None:
            u, v = (int(pixel[0]), int(pixel[1]))
            height, width = capture.depth_m.shape
            if not (0 <= u < width and 0 <= v < height):
                raise ValueError("target_pixel_uv is outside the live depth image.")
            selected[v, u] = True
            mode = "pixel"
            minimum_valid_points = 1
        else:
            x0, y0, x1, y1 = (int(value) for value in region)
            height, width = capture.depth_m.shape
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("pixel_region_xyxy is outside the live depth image.")
            if (x1 - x0) * (y1 - y0) > profile.maximum_region_pixels:
                raise ValueError(
                    "pixel_region_xyxy exceeds the reviewed pixel bound."
                )
            selected[y0:y1, x0:x1] = True
            mode = "region"
            minimum_valid_points = profile.minimum_region_valid_points
        geometry = project_depth_mask_geometry(
            capture,
            selected,
            minimum_valid_points=minimum_valid_points,
            maximum_depth_m=profile.maximum_depth_m,
            robust_lower_quantile=profile.robust_lower_quantile,
            robust_upper_quantile=profile.robust_upper_quantile,
            world_z_min_m=parameters["world_z_min_m"],
            world_z_max_m=parameters["world_z_max_m"],
            insufficient_points_error=(
                "back_projection_pixel_depth_invalid"
                if mode == "pixel"
                else "back_projection_region_insufficient_points"
            ),
        )
        return {
            "source_id": capture.source_id,
            "role": capture.role,
            "station_revision": capture.station_revision,
            "capture_group_id": capture.capture_group_id,
            "depth_content_hash": capture.depth_content_hash,
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "frame_graph_hash": profile.frame_graph_hash,
            "coordinate_mode": mode,
            "world_xyz_m": list(geometry.world_center_xyz_m),
            "depth_m": geometry.median_depth_m,
            "valid_point_count": geometry.valid_point_count,
        }


def _project_pixels(
    *,
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    intrinsics: Mapping[str, float | int],
    reference_from_camera: np.ndarray,
) -> np.ndarray:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    camera = np.column_stack(
        (
            (u - cx) * depth / fx,
            (v - cy) * depth / fy,
            depth,
        )
    )
    transform = reference_from_camera
    return camera @ transform[:3, :3].T + transform[:3, 3]


def build_back_projection_executor(
    *,
    registration: BackProjectionRegistration,
    provider: LiveBackProjectionProvider,
    current_station_revision: Callable[[], int],
    max_retained_operations: int,
    clock: Callable[[], float] = time.time,
) -> PassiveReadSkillExecutor:
    if provider.registration is not registration:
        raise ValueError("Back-projection provider and registration differ.")

    def read(request: Any, _operation: Any) -> PassiveReadResult:
        result = provider.project_request(request)
        if current_station_revision() != request.expected_station_revision:
            raise RuntimeError("Station revision changed during back-projection.")
        return PassiveReadResult(
            execution_result=result,
            evidence_refs=(
                "live-depth-capture:" + result["capture_group_id"],
                "live-depth-content:" + result["depth_content_hash"],
            ),
            reason_code="passive_depth_back_projection_completed",
        )

    return PassiveReadSkillExecutor(
        port_id=f"agent-gateway:{registration.profile.target_kind}-passive-back-projection:{registration.binding.runtime_session_id}",
        executor_id=registration.binding.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )


__all__ = [
    "BackProjectionRegistration",
    "CorrelatedPlacementCapture",
    "CorrelatedPlacementCapturePort",
    "LiveAlignedDepthCapture",
    "LiveAlignedDepthCapturePort",
    "LiveBackProjectionProvider",
    "ProjectedDepthGeometry",
    "build_back_projection_executor",
    "build_back_projection_registration",
    "project_depth_mask_geometry",
]

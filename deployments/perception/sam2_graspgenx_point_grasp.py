"""SAM2 + GraspGenX point-conditioned DMTAC M grasp proposals."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
from PIL import Image

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1,
)
from robot_motion.grasp_visible_scene import (
    PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER,
    ParallelJawSolidGeometry,
    VisibleGraspScene,
    estimate_grasp_surface_normals,
    parallel_jaw_capture_support,
    parallel_jaw_opposed_capture_support,
    parallel_jaw_pad_height_support,
    parallel_jaw_surface_alignment,
)

from .dmtac_grasp_overlay import (
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
    DMTAC_M_GRASPGENX_FINGER_LENGTH_M,
    DMTAC_M_GRASPGENX_FINGER_THICKNESS_M,
    DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_XYZ_M,
    DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_Z_M,
    DMTAC_M_GRASPGENX_MAX_OPENING_M,
    DMTAC_M_GRASPGENX_PAD_HEIGHT_M,
    DMTAC_M_GRASPGENX_PALM_Z_M,
    DMTAC_M_GRASPGENX_PROXY_FINGERTIP_EXTENSION_M,
    DMTAC_M_TOOL_TCP_FROM_FLANGE_Z_M,
    reviewed_graspgenx_dmtac_solid_geometry,
)
from .grasp_first_path import (
    FirstGraspPathPolicy,
    diverse_grasp_search_order,
    grasp_capture_height_acceptable,
    grasp_capture_support_cost,
    grasp_contact_height_observed,
)
from .grasp_planning_priority import (
    GRASP_PLANNING_ORDER_FIELD,
    GRASP_PLANNING_ORDER_SCOPE,
)
from .graspgenx_client import GraspGenXInferenceBackend
from .point_grasp_failure import PointGraspPlanningError
from .sam2_vgn_point_grasp import (
    DEFAULT_DMTAC_M_URDF,
    DEFAULT_DMTAC_M_URDF_SHA256,
    PointGraspCandidate,
    PointGraspResult,
    Sam2VgnPointGraspPlanner,
)

GRASPGENX_GIT_URL = "https://github.com/NVlabs/GraspGenX.git"
GRASPGENX_REVIEWED_REVISION = "d9464a263f2bcb7ad70b13d75de04cadde8d9ff7"
GRASPGENX_CHECKPOINT_GIT_URL = "https://huggingface.co/adithyamurali/GraspGenXModel"
GRASPGENX_CHECKPOINT_REVIEWED_REVISION = "7c834043c11a11417e31d6d5ea9355801e40a2c1"
GRASPGENX_ASSET_SCHEMA_V1 = "dexteleop.perception_graspgenx_point_grasp_assets.v1"
GRASPGENX_SOURCE_LICENSE = "Apache-2.0"
GRASPGENX_CHECKPOINT_LICENSE = "NVIDIA-Open-Model-License"

# DMTAC M's inner closing volume in the calibrated tool convention.  GraspGenX
# expects +Z approach and +X closing in the supplied model frame. The reviewed
# DMTAC origin is 29 mm distal to the nominal palm, not at that boundary. Both
# conditioning boxes and the OBB fingertip depth must therefore use the same
# axial interval as capture/collision checks; the backend does not rebase it.
# The linkage keeps opposing pads parallel, so this is a parallel_2f embodiment.
DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1: dict[str, object] = {
    "extents_open": (
        DMTAC_M_GRASPGENX_MAX_OPENING_M,
        DMTAC_M_GRASPGENX_PAD_HEIGHT_M,
        DMTAC_M_GRASPGENX_FINGER_LENGTH_M,
    ),
    "offset_open": (
        0.0, 0.0,
        DMTAC_M_GRASPGENX_PALM_Z_M + DMTAC_M_GRASPGENX_FINGER_LENGTH_M / 2.0,
    ),
    "extents_mid": (
        DMTAC_M_GRASPGENX_MAX_OPENING_M / 2.0,
        DMTAC_M_GRASPGENX_PAD_HEIGHT_M,
        DMTAC_M_GRASPGENX_FINGER_LENGTH_M,
    ),
    "offset_mid": (
        0.0, 0.0,
        DMTAC_M_GRASPGENX_PALM_Z_M + DMTAC_M_GRASPGENX_FINGER_LENGTH_M / 2.0,
    ),
    "gripper_type": 0,
    "fingertip_depth": (
        DMTAC_M_GRASPGENX_PALM_Z_M + DMTAC_M_GRASPGENX_FINGER_LENGTH_M
    ),
}
DMTAC_M_GRASPGENX_WIDTH_CLEARANCE_M = 0.006
DMTAC_M_GRASPGENX_TERMINAL_TARGET_CONTACT_DEPTH_M = 0.006
_DMTAC_GRASPGENX_IDENTITY_ROTATION = (
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
GRASPGENX_CANDIDATE_FAMILY_TRANSLATION_M = 0.015
GRASPGENX_CANDIDATE_FAMILY_APPROACH_DEG = 15.0
GRASPGENX_CANDIDATE_APPROACH_DIVERSITY_DEG = 20.0
GRASPGENX_DMTAC_REACHABILITY_ORIENTATION_VARIANTS_LOCAL_AXIS_DEG = (
    ("x", -15.0),
    ("x", -10.0),
    ("x", 10.0),
    ("x", 15.0),
    ("y", -20.0),
    ("y", 20.0),
    ("z", -30.0),
    ("z", 30.0),
)
GRASPGENX_DMTAC_ORIENTATION_VARIANT_SOURCE_LIMIT = 8
GRASPGENX_DMTAC_ORIENTATION_VARIANT_RETURN_LIMIT = 2
GRASPGENX_DMTAC_ORIENTATION_VARIANT_SCORE_DECAY_DEG = 180.0
GRASPGENX_DMTAC_ARM_SIDE_VARIANT_SOURCE_LIMIT = 32
GRASPGENX_DMTAC_ARM_SIDE_VARIANT_MAX_ROTATION_DEG = 100.0
GRASPGENX_DMTAC_ARM_SIDE_APPROACH_YAW_OFFSETS_DEG = (0.0, -20.0, 20.0)
GRASPGENX_DMTAC_TOP_APPROACH_COSINE = 0.7
GRASPGENX_DMTAC_SUPPORT_CLEARANCE_WORLD_Z_M = (0.005, 0.010, 0.015, 0.030)
# The saved DMTAC table scenes contain more than 200 distinct raw approach
# directions.  Sixty-four sources can still end with only two unique admitted
# poses because several low-collision sources describe the same physical pose.
# Keep the search bounded, but inspect enough distinct sources to populate the
# five-candidate UI without weakening the zero-background-collision rule.
GRASPGENX_DMTAC_SUPPORT_CLEARANCE_SOURCE_LIMIT = 128
GRASPGENX_DMTAC_SUPPORT_CLEARANCE_RETURN_LIMIT = 8
GRASPGENX_DMTAC_SUPPORT_CLEARANCE_SCORE_DECAY_M = 0.060
GRASPGENX_DMTAC_CENTERING_MAX_SHIFT_M = 0.015
GRASPGENX_DMTAC_CENTERING_SOURCE_LIMIT = 32
GRASPGENX_DMTAC_CENTERING_SCORE_DECAY_M = 0.060
GRASPGENX_DMTAC_CENTERING_POSE_SOURCE = "dmtac_centering_variant_v1"
GRASPGENX_DMTAC_AXIAL_POSE_SOURCE = "dmtac_axial_insertion_variant_v1"
GRASPGENX_DMTAC_AXIAL_SHIFTS_M = (0.005, 0.010, 0.015, 0.020)
GRASPGENX_DMTAC_AXIAL_SOURCE_LIMIT = 16
GRASPGENX_DMTAC_AXIAL_SOURCE_ORDER = "shallow_filter_then_bounded_motion_family_anchors_v2"
# The support score normalizes depth over 20% of the nominal finger length:
# 0.25 identifies a median observed insertion below 5 mm, not grasp success.
GRASPGENX_DMTAC_AXIAL_SHALLOW_SUPPORT = 0.25
GRASPGENX_DMTAC_HEIGHT_POSE_SOURCE = "dmtac_pad_height_centering_variant_v1"
GRASPGENX_DMTAC_HEIGHT_MAX_SHIFT_M = 0.040
GRASPGENX_DMTAC_HEIGHT_SOURCE_LIMIT = 32
GRASPGENX_DMTAC_HEIGHT_SHIFT_FRACTIONS = (0.5, 1.0)


def _pad_height_shifts(center_shift_m: float) -> tuple[float, ...]:
    """Bound each actual proposal, not the unrestricted optimum band centre."""
    return tuple(delta for fraction in GRASPGENX_DMTAC_HEIGHT_SHIFT_FRACTIONS
                 if 1e-9 < abs(delta := center_shift_m * fraction)
                 <= GRASPGENX_DMTAC_HEIGHT_MAX_SHIFT_M)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraspGenXCandidateSelection:
    """Select a bounded subset after physical admission and before publication."""

    selected_indices: tuple[int, ...]
    diagnostics: Mapping[str, int | float | str | None]


CandidateSelector = Callable[
    [tuple[PointGraspCandidate, ...], tuple[float, float, float], int],
    GraspGenXCandidateSelection,
]


@dataclass(frozen=True)
class GraspGenXComponentProposals:
    """Shared physical proposals with explicitly named, non-executable views."""

    candidate_pool: tuple[PointGraspCandidate, ...]
    results_by_component: Mapping[str, PointGraspResult]
    filter_summary_by_component: Mapping[str, Mapping[str, int | float | str | None]]
    selection_timing_ms_by_component: Mapping[str, float]


@dataclass(frozen=True)
class _GraspGenXCandidatePool:
    candidates: tuple[PointGraspCandidate, ...]
    generic_indices: tuple[int, ...]
    reserved_variant_indices: tuple[int, ...]
    summary: Mapping[str, int | float | str | None]
    replay_arrays: Mapping[str, np.ndarray] | None = None


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _lower_hex(value: object, *, field_name: str, length: int) -> str:
    text = _required(value, field_name=field_name)
    if len(text) != length or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(
            f"{field_name} must be {length} lowercase hexadecimal characters."
        )
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rigid_matrix(value: object, *, field_name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain 16 numeric values.") from exc
    if not np.all(np.isfinite(matrix)) or not np.allclose(
        matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8
    ):
        raise ValueError(f"{field_name} must be a finite homogeneous transform.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"{field_name} rotation must be right-handed and orthonormal.")
    return matrix


def _strict_rigid_matrix(value: object, *, field_name: str) -> np.ndarray:
    """Validate a pose against the downstream CoordinateFrameGraph contract."""

    matrix = _rigid_matrix(value, field_name=field_name)
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-9,
        rtol=0.0,
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9, rtol=0.0):
        raise ValueError(f"{field_name} rotation must satisfy the strict frame contract.")
    return matrix


def _normalize_graspgenx_rigid_matrix(
    value: object,
    *,
    field_name: str,
) -> np.ndarray:
    """Project one accepted float32 model pose to its nearest proper rotation."""

    matrix = _rigid_matrix(value, field_name=field_name)
    rotation = matrix[:3, :3]
    try:
        left, _singular_values, right_t = np.linalg.svd(rotation)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{field_name} rotation could not be normalized.") from exc
    projected = left @ right_t
    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right_t
    normalized = matrix.copy()
    normalized[:3, :3] = projected
    return _strict_rigid_matrix(normalized, field_name=field_name)


def _local_axis_rotation(axis: str, angle_deg: float) -> np.ndarray:
    """Return one grasp-local axis rotation as a homogeneous transform."""

    angle_rad = math.radians(float(angle_deg))
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    rotations = {
        "x": ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
        "y": ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        "z": ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
    }
    if axis not in rotations:
        raise ValueError("orientation variant axis must be x, y, or z")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotations[axis], dtype=np.float64)
    return transform


@dataclass(frozen=True)
class GraspGenXPointGraspModelSpec:
    model_profile_id: str
    bundle_root: Path
    source_revision: str
    checkpoint_revision: str
    generator_config_sha256: str
    generator_checkpoint_sha256: str
    discriminator_config_sha256: str
    discriminator_checkpoint_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_profile_id",
            _required(self.model_profile_id, field_name="model_profile_id"),
        )
        root = Path(self.bundle_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("bundle_root must be an existing directory.")
        object.__setattr__(self, "bundle_root", root)
        source_revision = _lower_hex(
            self.source_revision, field_name="source_revision", length=40
        )
        checkpoint_revision = _lower_hex(
            self.checkpoint_revision, field_name="checkpoint_revision", length=40
        )
        if source_revision != GRASPGENX_REVIEWED_REVISION:
            raise ValueError("GraspGenX source revision is not reviewed.")
        if checkpoint_revision != GRASPGENX_CHECKPOINT_REVIEWED_REVISION:
            raise ValueError("GraspGenX checkpoint revision is not reviewed.")
        for field_name in (
            "generator_config_sha256",
            "generator_checkpoint_sha256",
            "discriminator_config_sha256",
            "discriminator_checkpoint_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _lower_hex(getattr(self, field_name), field_name=field_name, length=64),
            )

    @classmethod
    def from_asset_bundle(cls, bundle_root: str | Path) -> GraspGenXPointGraspModelSpec:
        root = Path(bundle_root).expanduser().resolve(strict=True)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        expected_fields = {
            "schema_version",
            "model_profile_id",
            "source",
            "checkpoint_repository",
            "files",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected_fields:
            raise ValueError("GraspGenX asset manifest fields are invalid.")
        if manifest["schema_version"] != GRASPGENX_ASSET_SCHEMA_V1:
            raise ValueError("Unsupported GraspGenX asset manifest schema.")
        source = manifest["source"]
        if not isinstance(source, dict) or set(source) != {
            "git_url",
            "revision",
            "license",
        }:
            raise ValueError("GraspGenX source identity is invalid.")
        if (
            source["git_url"] != GRASPGENX_GIT_URL
            or source["license"] != GRASPGENX_SOURCE_LICENSE
        ):
            raise ValueError("GraspGenX source URL or license is not reviewed.")
        checkpoint_repository = manifest["checkpoint_repository"]
        if not isinstance(checkpoint_repository, dict) or set(
            checkpoint_repository
        ) != {
            "git_url",
            "revision",
            "license",
        }:
            raise ValueError("GraspGenX checkpoint repository identity is invalid.")
        if (
            checkpoint_repository["git_url"] != GRASPGENX_CHECKPOINT_GIT_URL
            or checkpoint_repository["license"] != GRASPGENX_CHECKPOINT_LICENSE
        ):
            raise ValueError("GraspGenX checkpoint URL or license is not reviewed.")
        files = manifest["files"]
        expected_files = {
            "generator_config",
            "generator_checkpoint",
            "discriminator_config",
            "discriminator_checkpoint",
        }
        if not isinstance(files, dict) or set(files) != expected_files:
            raise ValueError("GraspGenX manifest checkpoint file set is invalid.")
        hashes: dict[str, str] = {}
        expected_names = {
            "generator_config": "config.yaml",
            "generator_checkpoint": "epoch_736.pth",
            "discriminator_config": "config.yaml",
            "discriminator_checkpoint": "epoch_1056.pth",
        }
        for name, expected_name in expected_names.items():
            entry = files[name]
            if not isinstance(entry, dict) or set(entry) != {"relative_path", "sha256"}:
                raise ValueError(f"GraspGenX {name} entry is invalid.")
            relative = Path(
                _required(
                    entry["relative_path"], field_name=f"files.{name}.relative_path"
                )
            )
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.name != expected_name
            ):
                raise ValueError(f"GraspGenX {name} path is invalid.")
            path = (root / relative).resolve(strict=True)
            if not path.is_file() or root not in path.parents:
                raise ValueError(f"GraspGenX {name} must stay inside the asset bundle.")
            expected_sha256 = _lower_hex(
                entry["sha256"], field_name=f"files.{name}.sha256", length=64
            )
            if _sha256_file(path) != expected_sha256:
                raise ValueError(f"GraspGenX {name} hash mismatch.")
            hashes[name] = expected_sha256
        return cls(
            model_profile_id=manifest["model_profile_id"],
            bundle_root=root,
            source_revision=source["revision"],
            checkpoint_revision=checkpoint_repository["revision"],
            generator_config_sha256=hashes["generator_config"],
            generator_checkpoint_sha256=hashes["generator_checkpoint"],
            discriminator_config_sha256=hashes["discriminator_config"],
            discriminator_checkpoint_sha256=hashes["discriminator_checkpoint"],
        )

    @property
    def checkpoint_root(self) -> Path:
        return self.bundle_root / "checkpoints/release"

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "model_profile_id": self.model_profile_id,
                "source": {
                    "git_url": GRASPGENX_GIT_URL,
                    "revision": self.source_revision,
                    "license": GRASPGENX_SOURCE_LICENSE,
                },
                "checkpoint_repository": {
                    "git_url": GRASPGENX_CHECKPOINT_GIT_URL,
                    "revision": self.checkpoint_revision,
                    "license": GRASPGENX_CHECKPOINT_LICENSE,
                },
                "file_sha256": {
                    "generator_config": self.generator_config_sha256,
                    "generator_checkpoint": self.generator_checkpoint_sha256,
                    "discriminator_config": self.discriminator_config_sha256,
                    "discriminator_checkpoint": self.discriminator_checkpoint_sha256,
                },
            },
            domain="dexteleop.perception.graspgenx_point_grasp_model_spec.v1",
        )


@dataclass(frozen=True)
class GraspGenXSelectionCandidate(PointGraspCandidate):
    """Ephemeral pool lineage for selection, never a wire candidate identity."""
    symmetry_parent_index: int | None = None
    capture_support_score: float = 1.0
    axial_parent_index: int | None = None
    axial_shift_m: float = 0.0
    capture_height_support_score: float = 1.0
    capture_height_center_shift_m: float = 0.0
    height_parent_index: int | None = None
    height_shift_m: float = 0.0
    raw_index: int | None = None
    opposed_capture_support_score: float = 1.0
    surface_alignment_score: float = 1.0
    surface_alignment_observed_sides: int = 0
    surface_opposition_score: float = 0.0
    contact_height_support_score: float = 0.0
    contact_height_center_shift_m: float = 0.0
    contact_height_evidence_weight: float = 0.0
    height_shift_basis: str | None = None


@dataclass(frozen=True)
class _RankedGraspCandidate:
    score: float
    model_score: float
    click_distance_m: float
    gripper_width_m: float
    camera_from_grasp_geometry: np.ndarray
    base_from_grasp_geometry: np.ndarray
    pose_source: str = "model_scored"
    orientation_variant_axis: str | None = None
    orientation_variant_angle_deg: float = 0.0
    support_clearance_world_z_m: float = 0.0
    symmetry_parent: _RankedGraspCandidate | None = None
    capture_support_score: float = 1.0
    axial_parent: _RankedGraspCandidate | None = None
    axial_shift_m: float = 0.0
    capture_height_support_score: float = 1.0
    capture_height_center_shift_m: float = 0.0
    height_parent: _RankedGraspCandidate | None = None
    height_shift_m: float = 0.0
    raw_index: int | None = None
    opposed_capture_support_score: float = 1.0
    surface_alignment_score: float = 1.0
    surface_alignment_observed_sides: int = 0
    surface_opposition_score: float = 0.0
    contact_height_support_score: float = 0.0
    contact_height_center_shift_m: float = 0.0
    contact_height_evidence_weight: float = 0.0
    height_shift_basis: str | None = None


def _pose_is_duplicate(pose: np.ndarray, accepted: list[np.ndarray] | np.ndarray) -> bool:
    """Compare already validated poses with the existing allclose(rtol=0) rule."""
    if len(accepted) == 0:
        return False
    return bool(np.any(np.all(np.abs(np.asarray(accepted) - pose) <= 1e-6, axis=(1, 2))))


def _deduplicate_ranked_poses(candidates: list[_RankedGraspCandidate]) -> list[_RankedGraspCandidate]:
    # Keep the same greedy, order-dependent policy. A tolerance is not a
    # transitive equivalence relation, so rounding/hash bins would change it.
    retained: list[_RankedGraspCandidate] = []
    poses = np.empty((len(candidates), 4, 4), dtype=np.float64)
    for candidate in candidates:
        pose = candidate.camera_from_grasp_geometry
        if not _pose_is_duplicate(pose, poses[:len(retained)]):
            poses[len(retained)] = pose
            retained.append(candidate)
    return retained


class Sam2GraspGenXPointGraspPlanner:
    """Proposal-only GraspGenX planner using a retained aligned RGB-D frame."""

    backend_id = "graspgenx"

    def __init__(
        self,
        *,
        model_spec: GraspGenXPointGraspModelSpec,
        point_segmenter: object,
        client: GraspGenXInferenceBackend,
        planner: str = "graspmoe",
        num_grasps: int = 200,
        server_top_k: int = -1,
        score_threshold: float = -1.0,
        max_gripper_width_m: float = DMTAC_M_GRASPGENX_MAX_OPENING_M,
        max_click_distance_m: float = 0.100,
        solid_geometry: ParallelJawSolidGeometry | None = None,
        orientation_variants_local_axis_degrees: tuple[tuple[str, float], ...] = (),
        orientation_variant_source_limit: int = (
            GRASPGENX_DMTAC_ORIENTATION_VARIANT_SOURCE_LIMIT
        ),
        preferred_approach_origin_base_m: tuple[float, float, float] | None = None,
        preferred_approach_origins_base_m: tuple[tuple[float, float, float], ...] = (),
        enable_finger_exchange: bool = False,
        background_proxy_padding_m: float = 0.0,
    ) -> None:
        if not isinstance(model_spec, GraspGenXPointGraspModelSpec):
            raise TypeError("model_spec must be typed.")
        if not callable(getattr(point_segmenter, "segment_point", None)):
            raise TypeError("point_segmenter must expose segment_point().")
        if not isinstance(client, GraspGenXInferenceBackend):
            raise TypeError(
                "client must implement the GraspGenX inference backend contract."
            )
        if planner not in {"graspmoe", "diffusion"}:
            raise ValueError("planner must be 'graspmoe' or 'diffusion'.")
        if isinstance(num_grasps, bool) or not 1 <= int(num_grasps) <= 1_000:
            raise ValueError("num_grasps must be in [1, 1000].")
        if isinstance(server_top_k, bool) or not (
            int(server_top_k) == -1 or 1 <= int(server_top_k) <= 1_000
        ):
            raise ValueError("server_top_k must be -1 or in [1, 1000].")
        if not 0.0 < float(max_gripper_width_m) <= DMTAC_M_GRASPGENX_MAX_OPENING_M:
            raise ValueError(
                "DMTAC M GraspGenX gripper width must be in "
                f"(0, {DMTAC_M_GRASPGENX_MAX_OPENING_M:.4f}]."
            )
        if not 0.0 < float(max_click_distance_m) <= 0.200:
            raise ValueError("max_click_distance_m must be in (0, 0.200].")
        threshold = float(score_threshold)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("score_threshold must be finite and in [-1, 1].")
        self.model_spec = model_spec
        self.point_segmenter = point_segmenter
        self.client = client
        self.planner = planner
        self.num_grasps = int(num_grasps)
        self.server_top_k = int(server_top_k)
        self.score_threshold = threshold
        self.max_gripper_width_m = float(max_gripper_width_m)
        self.max_click_distance_m = float(max_click_distance_m)
        if (isinstance(background_proxy_padding_m, bool)
                or not math.isfinite(background_proxy_padding_m)
                or background_proxy_padding_m < 0):
            raise ValueError("background_proxy_padding_m must be finite and nonnegative.")
        self.background_proxy_padding_m = float(background_proxy_padding_m)
        if not isinstance(enable_finger_exchange, bool):
            raise TypeError("enable_finger_exchange must be bool.")
        self.enable_finger_exchange = enable_finger_exchange
        orientation_variants = []
        for value in orientation_variants_local_axis_degrees:
            if (
                not isinstance(value, (tuple, list))
                or len(value) != 2
                or not isinstance(value[0], str)
                or isinstance(value[1], bool)
            ):
                raise ValueError(
                    "orientation variants must be typed (axis, angle_deg) pairs."
                )
            try:
                orientation_variants.append((value[0], float(value[1])))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "orientation variant angles must be finite numbers."
                ) from error
        orientation_variants = tuple(orientation_variants)
        if (
            len(orientation_variants) > 8
            or len(set(orientation_variants)) != len(orientation_variants)
            or any(
                axis not in {"x", "y", "z"}
                or not math.isfinite(angle_deg)
                or abs(angle_deg) < 1e-9
                or abs(angle_deg) > 45.0
                for axis, angle_deg in orientation_variants
            )
        ):
            raise ValueError(
                "orientation_variants_local_axis_degrees must contain at most "
                "eight unique x/y/z axis-angle pairs with finite non-zero "
                "angles within [-45, 45]."
            )
        source_limit = int(orientation_variant_source_limit)
        if (
            isinstance(orientation_variant_source_limit, bool)
            or not 1 <= source_limit <= 32
        ):
            raise ValueError("orientation_variant_source_limit must be in [1, 32].")
        self.orientation_variants_local_axis_degrees = orientation_variants
        self.orientation_variant_source_limit = source_limit
        if preferred_approach_origin_base_m is not None and preferred_approach_origins_base_m:
            raise ValueError("Specify a single approach origin or multiple origins, not both.")
        origins = ((preferred_approach_origin_base_m,)
            if preferred_approach_origin_base_m is not None else preferred_approach_origins_base_m)
        if len(origins) > 2:
            raise ValueError("At most two approach origins are supported.")
        preferred_origins = []
        for origin in origins:
            preferred_origin_array = np.asarray(
                origin, dtype=np.float64
            )
            if preferred_origin_array.shape != (3,) or not np.all(
                np.isfinite(preferred_origin_array)
            ):
                raise ValueError(
                    "Each approach origin must be three finite values."
                )
            preferred_origins.append(tuple(float(value) for value in preferred_origin_array))
        self.preferred_approach_origins_base_m = tuple(dict.fromkeys(preferred_origins))
        expected_collision_geometry = ParallelJawSolidGeometry(
            DMTAC_M_GRASPGENX_PALM_Z_M,
            DMTAC_M_GRASPGENX_FINGER_LENGTH_M + DMTAC_M_GRASPGENX_PROXY_FINGERTIP_EXTENSION_M,
            DMTAC_M_GRASPGENX_FINGER_THICKNESS_M,
            DMTAC_M_GRASPGENX_PAD_HEIGHT_M,
            self.max_gripper_width_m,
        )
        self._solid_geometry = (
            solid_geometry
            if solid_geometry is not None
            else reviewed_graspgenx_dmtac_solid_geometry(
                maximum_opening_m=self.max_gripper_width_m
            )
        )
        if not isinstance(self._solid_geometry, ParallelJawSolidGeometry):
            raise TypeError("solid_geometry must be typed")
        if (
            replace(
                self._solid_geometry,
                tool_from_solid_xyz_m=(0, 0, 0),
                solid_yaw_rad=0,
                tool_from_solid_rotation=_DMTAC_GRASPGENX_IDENTITY_ROTATION,
            )
            != expected_collision_geometry
        ):
            raise ValueError(
                "GraspGenX adapter requires the reviewed nominal DMTAC M dimensions"
            )
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._last_target_mask: np.ndarray | None = None
        self._last_plan_timing_ms: dict[str, float] = {}
        self._last_filter_summary: dict[str, int | float | str | None] = {}

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "model_spec_hash": self.model_spec.semantic_content_hash,
                "sam2_segmenter_hash": _required(
                    getattr(self.point_segmenter, "semantic_content_hash", None),
                    field_name="point_segmenter.semantic_content_hash",
                ),
                "client": self.client.semantic_config,
                "planner": self.planner,
                "num_grasps": self.num_grasps,
                "server_top_k": self.server_top_k,
                "score_threshold": self.score_threshold,
                "inference_frame": "gravity_aligned_reference_z_up",
                "candidate_family_nms": {
                    "translation_m": GRASPGENX_CANDIDATE_FAMILY_TRANSLATION_M,
                    "approach_axis_deg": GRASPGENX_CANDIDATE_FAMILY_APPROACH_DEG,
                    "oriented_closing_axis_deg": (
                        GRASPGENX_CANDIDATE_FAMILY_APPROACH_DEG
                    ),
                },
                "candidate_selection": {
                    "policy": "collision_readmitted_arm_motion_shortlist_v11",
                    "approach_axis_deg": (
                        GRASPGENX_CANDIDATE_APPROACH_DIVERSITY_DEG
                    ),
                    "fallback": "score_ordered_scored_pose_variants",
                    "duplicate_pose_atol": 1e-6,
                    "dmtac_orientation_variants": {
                        "axis_angle_deg": self.orientation_variants_local_axis_degrees,
                        "frame": "grasp_geometry_local",
                        "source_limit": self.orientation_variant_source_limit,
                        "return_limit": (
                            GRASPGENX_DMTAC_ORIENTATION_VARIANT_RETURN_LIMIT
                        ),
                        "score_policy": "parent_score_exp_abs_angle_over_180",
                        "physical_readmission": (
                            "width_target_capture_terminal_collision_and_approach"
                        ),
                        "embodied_half_turn": ("z_180_about_observed_target_center"
                            if self.enable_finger_exchange else "disabled"),
                    },
                    "dmtac_centering_variants": {
                        "default_enabled": False,
                        "request_parameter": "refine_grasp_centers",
                        "pose_source": GRASPGENX_DMTAC_CENTERING_POSE_SOURCE,
                        "frame_axis": "grasp_geometry_local_x",
                        "minimum_abs_center_exclusive_m": DMTAC_M_GRASPGENX_WIDTH_CLEARANCE_M,
                        "maximum_abs_center_inclusive_m": GRASPGENX_DMTAC_CENTERING_MAX_SHIFT_M,
                        "source_limit": GRASPGENX_DMTAC_CENTERING_SOURCE_LIMIT,
                        "source_order": "distinct_approach_cones_then_proposal_priority_else_raw_model_score",
                        "source_requires": "raw_width_valid_and_yz_capture_overlap",
                        "score_decay_m": GRASPGENX_DMTAC_CENTERING_SCORE_DECAY_M,
                        "physical_readmission": "existing_prepare_pose_and_visible_scene",
                    },
                    "dmtac_axial_variants": {
                        "default_enabled": False,
                        "request_parameter": "refine_grasp_axial_depth",
                        "pose_source": GRASPGENX_DMTAC_AXIAL_POSE_SOURCE,
                        "frame_axis": "grasp_geometry_local_positive_z",
                        "shifts_m": GRASPGENX_DMTAC_AXIAL_SHIFTS_M,
                        "source_limit": GRASPGENX_DMTAC_AXIAL_SOURCE_LIMIT,
                        "shallow_support_exclusive": GRASPGENX_DMTAC_AXIAL_SHALLOW_SUPPORT,
                        "source_order": GRASPGENX_DMTAC_AXIAL_SOURCE_ORDER,
                        "return_per_source": 1,
                        "score_policy": "inherited_parent_not_neurally_rescored",
                        "physical_readmission": "existing_prepare_pose_and_visible_scene",
                    },
                    "dmtac_pad_height_variants": {
                        "enabled_with": "refine_grasp_centers",
                        "pose_source": GRASPGENX_DMTAC_HEIGHT_POSE_SOURCE,
                        "frame_axis": "grasp_geometry_local_y_not_station_z",
                        "maximum_abs_shift_m": GRASPGENX_DMTAC_HEIGHT_MAX_SHIFT_M,
                        "source_limit": GRASPGENX_DMTAC_HEIGHT_SOURCE_LIMIT,
                        "fractions": GRASPGENX_DMTAC_HEIGHT_SHIFT_FRACTIONS,
                        "minimum_support": FirstGraspPathPolicy().minimum_capture_height_support_score,
                        "source_order": "distinct_motion_families_of_low_density_or_contact_band_support",
                        "contact_band": "normal_weighted_local_pad_kernel_advisory_v1",
                        "contact_orientation_recovery": "bounded_height_sources_then_existing_orientation_variants_v1",
                        "contact_orientation_source_limit": min(self.orientation_variant_source_limit,
                            GRASPGENX_DMTAC_ORIENTATION_VARIANT_SOURCE_LIMIT),
                        "physical_readmission": "existing_prepare_pose_and_visible_scene",
                    },
                    "dmtac_arm_side_variants": {
                        "preferred_approach_origin_base_m": (
                            self.preferred_approach_origins_base_m[0]
                            if len(self.preferred_approach_origins_base_m) == 1 else None
                        ),
                        **({"preferred_approach_origins_base_m": self.preferred_approach_origins_base_m}
                           if len(self.preferred_approach_origins_base_m) > 1 else {}),
                        "source_limit": GRASPGENX_DMTAC_ARM_SIDE_VARIANT_SOURCE_LIMIT,
                        "maximum_rotation_deg": (
                            GRASPGENX_DMTAC_ARM_SIDE_VARIANT_MAX_ROTATION_DEG
                        ),
                        "top_approach_cosine": GRASPGENX_DMTAC_TOP_APPROACH_COSINE,
                        "side_approach_yaw_offsets_deg": (
                            GRASPGENX_DMTAC_ARM_SIDE_APPROACH_YAW_OFFSETS_DEG
                        ),
                        "side_grasp_axes": (
                            "horizontal_closing_vertical_height"
                        ),
                        "physical_readmission": (
                            "width_target_capture_terminal_collision_and_approach"
                        ),
                    },
                    "dmtac_support_clearance_variants": {
                        "world_z_offsets_m": (
                            GRASPGENX_DMTAC_SUPPORT_CLEARANCE_WORLD_Z_M
                        ),
                        "frame": "gravity_aligned_reference",
                        "source_limit": (
                            GRASPGENX_DMTAC_SUPPORT_CLEARANCE_SOURCE_LIMIT
                        ),
                        "return_limit": (
                            GRASPGENX_DMTAC_SUPPORT_CLEARANCE_RETURN_LIMIT
                        ),
                        "score_policy": "parent_score_exp_clearance_over_0_060m",
                        "source_condition": (
                            "terminal_blocked_then_admitted_shortlist_fallback"
                        ),
                        "physical_readmission": (
                            "width_target_capture_click_terminal_collision_and_approach"
                        ),
                    },
                },
                "gripper": {
                    "profile": "DMTAC_M",
                    "graspgenx_alignment_profile": DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
                    "solid_geometry": asdict(self.solid_geometry),
                    "sweep_volume_v1": DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
                    "model_base_from_output_z_m": 0.0,
                    "maximum_opening_m": self.max_gripper_width_m,
                    "width_clearance_m": DMTAC_M_GRASPGENX_WIDTH_CLEARANCE_M,
                    "capture_support_rule": "nonempty_joint_closing_window_v1",
                    "capture_support_preference": "median_axial_inset_over_0_2_finger_length",
                    "capture_height_support": "triangular_pad_kernel_over_best_local_height_band_v1",
                    "surface_alignment_preference": "single_face_credit_plus_separated_surface_bonus_v2",
                    "visible_collision_opening_basis": "maximum_opening",
                    "terminal_target_contact_rule": "inner_tactile_pad_band_only",
                    "terminal_target_contact_depth_m": (
                        DMTAC_M_GRASPGENX_TERMINAL_TARGET_CONTACT_DEPTH_M
                    ),
                    "visible_collision_scope": (
                        "target_surface_terminal_proxy_plus_optional_"
                        "capture_filtered_background_terminal_proxy"
                    ),
                    "background_collision": (
                        "optional_terminal_prefilter_then_motion_planner"
                    ),
                    "background_proxy_padding_m": self.background_proxy_padding_m,
                    "terminal_background_prefilter": (
                        "runtime_visible_scene_factory"
                    ),
                    "visible_depth_filter_hash": (
                        PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER.semantic_content_hash
                    ),
                    "maximum_visible_collision_points": 0,
                    "visible_endpoint_opening_m": self.max_gripper_width_m,
                    "unscored_pose_translation_allowed": True,
                    "variant_score_semantics": (
                        "parent_model_score_with_pose_delta_penalty_v1"
                    ),
                    "visible_approach_sweep_distance_m": 0.060,
                    "visible_approach_opening_m": self.max_gripper_width_m,
                    "grasp_center_from_tactile_xyz_m": list(
                        DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_XYZ_M
                    ),
                    "grasp_center_from_tactile_z_m": DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_Z_M,
                    "model_palm_z_m": DMTAC_M_GRASPGENX_PALM_Z_M,
                    "finger_length_m": DMTAC_M_GRASPGENX_FINGER_LENGTH_M,
                    "finger_thickness_m": DMTAC_M_GRASPGENX_FINGER_THICKNESS_M,
                    "pad_height_m": DMTAC_M_GRASPGENX_PAD_HEIGHT_M,
                    "tool_tcp_from_flange_m": [
                        0.0,
                        0.0,
                        DMTAC_M_TOOL_TCP_FROM_FLANGE_Z_M,
                    ],
                    "tcp_orientation_selection": "preserve_scored_model_orientation_v1",
                    "finger_exchange_deduplication": False,
                },
                "simulation_urdf": DEFAULT_DMTAC_M_URDF,
                "simulation_urdf_sha256": DEFAULT_DMTAC_M_URDF_SHA256,
            },
            domain="dexteleop.perception.sam2_graspgenx_point_grasp_planner.v1",
        )

    @property
    def last_target_mask(self) -> np.ndarray | None:
        with self._lock:
            return (
                None
                if self._last_target_mask is None
                else self._last_target_mask.copy()
            )

    @property
    def last_plan_timing_ms(self) -> dict[str, float]:
        with self._lock:
            return dict(self._last_plan_timing_ms)

    @property
    def last_filter_summary(self) -> dict[str, int | float | str | None]:
        with self._lock:
            return dict(self._last_filter_summary)

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Closed point-grasp planner cannot restart.")
            if self._started:
                return
            self.client.start()
            self._started = True

    def close(self) -> None:
        with self._lock:
            self.client.close()
            self._started = False
            self._closed = True

    def plan(
        self, image: Image.Image, depth_m: np.ndarray, *, pixel_uv: tuple[int, int],
        intrinsics: Mapping[str, float | int], base_from_camera: np.ndarray,
        max_candidates: int,
        visible_scene_factory: Callable[..., VisibleGraspScene] | None = None,
        use_visible_target_points: bool = False,
        candidate_selector: CandidateSelector | None = None,
        proposal_priority: Callable | None = None,
        refine_grasp_centers: bool = False,
        refine_grasp_axial_depth: bool = False,
        preparation_guard: Callable[[], None] | None = None,
    ) -> PointGraspResult:
        if candidate_selector is not None and not callable(candidate_selector):
            raise TypeError("candidate_selector must be callable or None.")
        observation, pool = self._prepare_pool(
            image, depth_m, pixel_uv=pixel_uv, intrinsics=intrinsics,
            base_from_camera=base_from_camera, max_candidates=max_candidates,
            visible_scene_factory=visible_scene_factory,
            use_visible_target_points=use_visible_target_points,
            for_embodiment=candidate_selector is not None,
            proposal_priority=proposal_priority,
            **({"preparation_guard": preparation_guard} if preparation_guard is not None else {}),
            refine_grasp_centers=refine_grasp_centers,
            refine_grasp_axial_depth=refine_grasp_axial_depth)
        began = time.perf_counter()
        try:
            candidates, summary = self._publish_pool(pool,
                target=observation.target_point_base_m, maximum=int(max_candidates),
                selector=candidate_selector)
            with self._lock:
                self._last_filter_summary = summary
            if not candidates:
                self._raise_no_candidates(summary, replay_arrays=pool.replay_arrays)
            return replace(observation, candidates=candidates)
        finally:
            self._finish_selection_timing(began, embodied=candidate_selector is not None)

    def plan_for_components(
        self, image: Image.Image, depth_m: np.ndarray, *, pixel_uv: tuple[int, int],
        intrinsics: Mapping[str, float | int], base_from_camera: np.ndarray,
        max_candidates: int, candidate_selectors: Mapping[str, CandidateSelector],
        visible_scene_factory: Callable[..., VisibleGraspScene] | None = None,
        use_visible_target_points: bool = False,
    ) -> GraspGenXComponentProposals:
        """Generate once for components sharing the same gripper and scene.

        Selectors retain their own arm-specific native IK/local-chain checks.
        An empty component view means no admitted candidate in its budget, not
        a perception failure or permission to switch arms during execution.
        """
        selectors = dict(candidate_selectors)
        if not selectors or len(selectors) > 2:
            raise ValueError("Provide one or two named component selectors.")
        for name, selector in selectors.items():
            _required(name, field_name="component name")
            if not callable(selector):
                raise TypeError("Each component selector must be callable.")
        observation, pool = self._prepare_pool(
            image, depth_m, pixel_uv=pixel_uv, intrinsics=intrinsics,
            base_from_camera=base_from_camera, max_candidates=max_candidates,
            visible_scene_factory=visible_scene_factory,
            use_visible_target_points=use_visible_target_points, for_embodiment=True)
        results, summaries, durations = {}, {}, {}
        began = time.perf_counter()
        try:
            for name, selector in selectors.items():
                selection_started = time.perf_counter()
                try:
                    candidates, summary = self._publish_pool(pool,
                        target=observation.target_point_base_m,
                        maximum=int(max_candidates), selector=selector)
                finally:
                    durations[name] = (time.perf_counter() - selection_started) * 1000.
                results[name] = replace(observation, candidates=candidates)
                summaries[name] = MappingProxyType(summary)
            return GraspGenXComponentProposals(
                candidate_pool=pool.candidates,
                results_by_component=MappingProxyType(results),
                filter_summary_by_component=MappingProxyType(summaries),
                selection_timing_ms_by_component=MappingProxyType(durations))
        finally:
            # last_filter_summary remains the common physical funnel. Do not
            # label the last evaluated arm's admission as a shared conclusion.
            self._finish_selection_timing(began, embodied=True)

    def _finish_selection_timing(self, began: float, *, embodied: bool) -> None:
        elapsed = (time.perf_counter() - began) * 1000.
        with self._lock:
            if embodied:
                self._last_plan_timing_ms["arm_candidate_selection"] = elapsed
            else:
                self._last_plan_timing_ms["candidate_filtering_excluding_arm_selection"] += elapsed
            self._last_plan_timing_ms["candidate_ranking_and_visible_collision"] += elapsed
            self._last_plan_timing_ms["plan_total"] += elapsed

    @staticmethod
    def _raise_no_candidates(summary: Mapping[str, object], *,
                             replay_arrays: Mapping | None = None) -> None:
        summary_json = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        _LOGGER.error("No point-grasp candidate survived: filter_summary=%s", summary_json)
        raise PointGraspPlanningError(summary, replay_arrays=replay_arrays)

    def _prepare_pool(
        self,
        image: Image.Image,
        depth_m: np.ndarray,
        *,
        pixel_uv: tuple[int, int],
        intrinsics: Mapping[str, float | int],
        base_from_camera: np.ndarray,
        max_candidates: int,
        visible_scene_factory: Callable[..., VisibleGraspScene] | None = None,
        use_visible_target_points: bool = False,
        for_embodiment: bool,
        proposal_priority: Callable | None = None,
        refine_grasp_centers: bool = False,
        refine_grasp_axial_depth: bool = False,
        preparation_guard: Callable[[], None] | None = None,
    ) -> tuple[PointGraspResult, _GraspGenXCandidatePool]:
        if preparation_guard is not None and not callable(preparation_guard):
            raise TypeError("preparation_guard must be callable or None.")
        if not isinstance(refine_grasp_centers, bool):
            raise TypeError("refine_grasp_centers must be bool.")
        if not isinstance(refine_grasp_axial_depth, bool):
            raise TypeError("refine_grasp_axial_depth must be bool.")
        started = time.perf_counter()
        timing: dict[str, float] = {}
        with self._lock:
            self._last_filter_summary = {}
            self._last_plan_timing_ms = {}
            if not self._started or self._closed:
                raise RuntimeError("Point-grasp planner is not running.")

        if preparation_guard is not None:
            preparation_guard()
        def finish_stage(name, began):
            timing[name] = (time.perf_counter() - began) * 1000.
            timing["plan_total"] = (time.perf_counter() - started) * 1000.
            with self._lock:
                self._last_plan_timing_ms = dict(timing)
        rgb = image.convert("RGB")
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.shape != (rgb.height, rgb.width) or not np.all(np.isfinite(depth)):
            raise ValueError("Depth must be a finite aligned HxW meter image.")
        if isinstance(max_candidates, bool) or not 1 <= int(max_candidates) <= 5:
            raise ValueError("max_candidates must be in [1, 5].")
        if visible_scene_factory is not None and not callable(visible_scene_factory):
            raise TypeError("visible_scene_factory must be callable or None.")
        u, v = pixel_uv
        if not (0 <= u < rgb.width and 0 <= v < rgb.height):
            raise ValueError("pixel_uv is outside the image.")
        values = {name: float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy")}
        if (
            int(intrinsics["width"]) != rgb.width
            or int(intrinsics["height"]) != rgb.height
        ):
            raise ValueError(
                "Intrinsics dimensions differ from the aligned RGB-D image."
            )
        if (
            values["fx"] <= 0.0
            or values["fy"] <= 0.0
            or not all(math.isfinite(value) for value in values.values())
        ):
            raise ValueError("Camera intrinsics are invalid.")
        transform_base_camera = _rigid_matrix(
            base_from_camera, field_name="base_from_camera"
        )
        if preparation_guard is not None:
            preparation_guard()
        stage_started = time.perf_counter()
        try:
            mask = np.asarray(
                self.point_segmenter.segment_point(rgb, pixel_uv=(u, v)), dtype=bool)
        finally:
            finish_stage("sam2_inference", stage_started)
        if preparation_guard is not None:
            preparation_guard()
        if mask.shape != depth.shape or not mask[v, u]:
            raise RuntimeError("SAM2 returned an invalid point mask.")
        with self._lock:
            self._last_target_mask = mask.copy()
        target_points_camera = Sam2VgnPointGraspPlanner._masked_points_camera(
            depth, mask, values
        ).astype(np.float32)
        if len(target_points_camera) > 50_000:
            indices = np.linspace(
                0, len(target_points_camera) - 1, 50_000, dtype=np.int64
            )
            target_points_camera = np.ascontiguousarray(
                target_points_camera[indices]
            )
        target_points_base = np.ascontiguousarray(
            target_points_camera @ transform_base_camera[:3, :3].T
            + transform_base_camera[:3, 3],
            dtype=np.float32,
        )
        target_depth = Sam2VgnPointGraspPlanner._target_depth(depth, mask, u=u, v=v)
        point_camera = np.asarray(
            [
                (u - values["cx"]) * target_depth / values["fx"],
                (v - values["cy"]) * target_depth / values["fy"],
                target_depth,
                1.0,
            ],
            dtype=np.float64,
        )
        point_base = transform_base_camera @ point_camera
        visible_scene = None
        if visible_scene_factory is not None:
            stage_started = time.perf_counter()
            try:
                visible_scene = visible_scene_factory(
                    depth_m=depth, target_mask=mask, intrinsics=values,
                    geometry=replace(self.solid_geometry, tool_from_solid_xyz_m=(0, 0, 0),
                        solid_yaw_rad=0, tool_from_solid_rotation=_DMTAC_GRASPGENX_IDENTITY_ROTATION))
            finally:
                finish_stage("visible_scene_construction", stage_started)
            if not isinstance(visible_scene, VisibleGraspScene):
                raise TypeError(
                    "visible_scene_factory must return a VisibleGraspScene."
                )
        if use_visible_target_points:
            if visible_scene is None:
                raise ValueError("Fused target inference requires its matching visible scene")
            target_points_camera = np.ascontiguousarray(visible_scene.points[visible_scene.target_mask], dtype=np.float32)
            if len(target_points_camera) < 3:
                raise RuntimeError("Fused scene contains insufficient target points")
            target_points_base = np.ascontiguousarray(
                target_points_camera @ transform_base_camera[:3, :3].T + transform_base_camera[:3, 3], dtype=np.float32)
        if preparation_guard is not None:
            preparation_guard()
        stage_started = time.perf_counter()
        try:
            grasps, scores = self.client.infer_object(
                # OBB top is gravity aligned; fuse before this single invocation.
                target_points_base, sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
                planner=self.planner, num_grasps=self.num_grasps,
                score_threshold=self.score_threshold, top_k=self.server_top_k)
        finally:
            finish_stage("graspgenx_inference", stage_started)
        if preparation_guard is not None:
            preparation_guard()
        stage_started = time.perf_counter()
        filter_summary: dict[str, int | float | str | None] = {}
        if (
            visible_scene is not None
            and visible_scene.depth_preprocessing_evidence is not None
        ):
            depth_evidence = visible_scene.depth_preprocessing_evidence
            filter_summary.update(
                {
                    "depth_filter_config_hash": str(
                        depth_evidence["config_hash"]
                    ),
                    "depth_filter_input_valid_points": int(
                        depth_evidence["input_valid_point_count"]
                    ),
                    "depth_filter_output_valid_points": int(
                        depth_evidence["output_valid_point_count"]
                    ),
                    "depth_filter_removed_isolated_background_points": int(
                        depth_evidence["removed_isolated_background_points"]
                    ),
                    "depth_filter_removed_target_points": int(
                        depth_evidence["removed_target_points"]
                    ),
                }
            )
        try:
            pool = self._rank_pool(
                grasps, scores, target_points_camera=target_points_camera,
                target_camera=point_camera[:3], base_from_camera=transform_base_camera,
                depth=depth, intrinsics=values, maximum=int(max_candidates),
                filter_summary=filter_summary, visible_scene=visible_scene,
                for_embodiment=for_embodiment, proposal_priority=proposal_priority,
                refine_grasp_centers=refine_grasp_centers,
                refine_grasp_axial_depth=refine_grasp_axial_depth,
                candidate_timing_ms=timing)
        finally:
            elapsed = (time.perf_counter() - stage_started) * 1000.
            # Preserve the inclusive legacy field; expose its exclusive child
            # so expensive arm checks are not misattributed to point collisions.
            timing["candidate_filtering_excluding_arm_selection"] = max(
                0., elapsed - timing.get("arm_candidate_selection", 0.))
            finish_stage("candidate_ranking_and_visible_collision", stage_started)
            with self._lock:
                self._last_filter_summary = dict(filter_summary)
        # These arrays belong to this request only. Retain the real segmented
        # mask and pre-filter proposals if a later arm check rejects the pool;
        # the pre-inference input deliberately has no valid target mask.
        replay_arrays = {
            "target_mask": mask.copy(),
            "raw_grasps_reference": np.asarray(grasps).copy(),
            "raw_scores": np.asarray(scores).copy(),
            "pool_tool_poses_reference": np.asarray(
                [candidate.base_from_tcp for candidate in pool.candidates],
                dtype=np.float64).reshape(-1, 4, 4),
            "pool_scores": np.asarray([candidate.score for candidate in pool.candidates]),
        }
        if visible_scene is not None:
            replay_arrays.update(scene_points_camera=visible_scene.points.copy(),
                                 scene_target_mask=visible_scene.target_mask.copy())
        pool = replace(pool, replay_arrays=replay_arrays)
        if not pool.candidates:
            self._raise_no_candidates(filter_summary, replay_arrays=replay_arrays)
        rows, columns = np.nonzero(mask)
        return PointGraspResult(
            target_point_camera_m=tuple(float(value) for value in point_camera[:3]),
            target_point_base_m=tuple(float(value) for value in point_base[:3]),
            target_mask_bbox_xyxy=(
                int(columns.min()),
                int(rows.min()),
                int(columns.max()) + 1,
                int(rows.max()) + 1,
            ),
            candidates=(),
        ), pool

    def _rank_pool(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        *,
        target_points_camera: np.ndarray,
        target_camera: np.ndarray,
        base_from_camera: np.ndarray,
        depth: np.ndarray,
        intrinsics: Mapping[str, float],
        maximum: int,
        filter_summary: dict[str, int | float | str | None] | None = None,
        visible_scene: VisibleGraspScene | None = None,
        for_embodiment: bool = False,
        proposal_priority: Callable | None = None,
        refine_grasp_centers: bool = False,
        refine_grasp_axial_depth: bool = False,
        candidate_timing_ms: dict[str, float] | None = None,
    ) -> _GraspGenXCandidatePool:
        internal_candidate_target = (
            max(maximum, 24) if for_embodiment else maximum
        )
        model_base_from_tcp = np.eye(4, dtype=np.float64)
        # The saved DMTAC alignment defines GraspGenX's output as the model
        # base itself.  Its palm boundary is model-local ``palm_z_m``; do not
        # apply the legacy 42.999 mm TCP correction a second time.
        camera_from_base = np.linalg.inv(base_from_camera)
        # Target fit and fixed ingress remain target-only checks. A live real
        # deployment may additionally provide the capture-filtered full scene;
        # use that scene only for the terminal grasp proxy. This lets the
        # generator remove candidates whose final gripper solids already hit a
        # visible background point, while the motion planner still owns the
        # full approach sweep and global path search.
        grasp_geometry = replace(
            self.solid_geometry,
            tool_from_solid_xyz_m=(0, 0, 0),
            solid_yaw_rad=0,
            tool_from_solid_rotation=_DMTAC_GRASPGENX_IDENTITY_ROTATION,
        )
        capture_geometry = replace(
            grasp_geometry, finger_length_m=DMTAC_M_GRASPGENX_FINGER_LENGTH_M)
        # Fast-search quality only. Reuse one request-local normal estimate;
        # ordinary candidate ordering and collision point arrays stay intact.
        surface_normals = surface_confidence = None
        normal_started = time.perf_counter()
        try:
            if refine_grasp_centers:
                surface_normals, surface_confidence = estimate_grasp_surface_normals(
                    target_points_camera, radius_m=capture_geometry.pad_height_m / 2)
        finally:
            if candidate_timing_ms is not None:
                candidate_timing_ms["candidate_detail_surface_normals"] = (
                    time.perf_counter() - normal_started) * 1000.
        target_scene = VisibleGraspScene(
            points=np.asarray(target_points_camera, dtype=np.float64),
            target_mask=np.ones(len(target_points_camera), dtype=np.bool_),
            # Inference and capture checks use the +X-closing model geometry
            # frame. Convert to the real controller TCP only at the output.
            geometry=grasp_geometry,
        )
        terminal_scene = target_scene
        if visible_scene is not None:
            if visible_scene.geometry != grasp_geometry:
                raise ValueError(
                    "visible_scene geometry must use the GraspGenX grasp frame."
                )
            terminal_scene = visible_scene
        ranked: list[_RankedGraspCandidate] = []
        counts: dict[str, int | float | str | None] = {
            "detected": 0,
            "width_valid": 0,
            "target_capture_valid": 0,
            "target_capture_joint_empty_rejected": 0,
            "click_distance_valid": 0,
            "visible_collision_valid": 0,
            "visible_approach_valid": 0,
            "visible_approach_collision_candidates": 0,
            "visible_approach_target_collision_candidates": 0,
            "visible_approach_background_collision_candidates": 0,
            "target_solid_collision_rejected": 0,
            "background_solid_collision_rejected": 0,
            "width_valid_downward_approach_count": 0,
            "target_capture_downward_approach_count": 0,
            "downward_approach_count": 0,
            "min_width_valid_downward_approach_cosine": None,
            "max_width_valid_downward_approach_cosine": None,
            "min_downward_approach_cosine": None,
            "max_downward_approach_cosine": None,
            "min_target_capture_violation_m": None,
            "closest_target_center_tcp_x_m": None,
            "closest_target_center_tcp_y_m": None,
            "closest_target_center_tcp_z_m": None,
            "min_downward_target_capture_violation_m": None,
            "closest_downward_target_center_tcp_x_m": None,
            "closest_downward_target_center_tcp_y_m": None,
            "closest_downward_target_center_tcp_z_m": None,
            "capture_palm_z_m": DMTAC_M_GRASPGENX_PALM_Z_M,
            "capture_fingertip_z_m": (
                DMTAC_M_GRASPGENX_PALM_Z_M + DMTAC_M_GRASPGENX_FINGER_LENGTH_M
            ),
            "capture_half_pad_height_m": DMTAC_M_GRASPGENX_PAD_HEIGHT_M / 2.0,
            "capture_coordinate_frame": "grasp_geometry",
            "collision_opening_m": self.max_gripper_width_m,
            "graspgenx_alignment_profile": DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
            "visible_collision_scope": (
                "target_surface_terminal_proxy_plus_capture_filtered_"
                "background_terminal_proxy"
                if visible_scene is not None
                else "target_surface_grasp_proxy_only"
            ),
            "background_collision": (
                "terminal_prefiltered_after_capture_self_filter"
                if visible_scene is not None
                else "deferred_to_motion_planner"
            ),
            "terminal_background_prefilter_enabled": visible_scene is not None,
            "background_proxy_padding_m": self.background_proxy_padding_m,
            "terminal_background_collision_rejected": 0,
            "visible_scene_point_count": len(terminal_scene.points),
            "visible_scene_background_point_count": int(
                np.count_nonzero(~terminal_scene.target_mask)
            ),
            "visible_collision_opening_basis": "maximum_opening",
            "visible_collision_opening_m": self.max_gripper_width_m,
            "terminal_target_contact_rule": "inner_tactile_pad_band_only",
            "terminal_target_contact_depth_m": (
                DMTAC_M_GRASPGENX_TERMINAL_TARGET_CONTACT_DEPTH_M
            ),
            "terminal_target_contact_points": 0,
            "min_visible_collision_opening_m": None,
            "max_visible_collision_opening_m": None,
            "orientation_variant_attempted": 0,
            "orientation_variant_width_valid": 0,
            "orientation_variant_target_capture_valid": 0,
            "orientation_variant_click_distance_valid": 0,
            "orientation_variant_visible_collision_valid": 0,
            "orientation_variant_visible_approach_valid": 0,
            "orientation_variant_visible_approach_collision_candidates": 0,
            "orientation_variant_target_solid_collision_rejected": 0,
            "orientation_variant_background_solid_collision_rejected": 0,
            "orientation_variant_selected": 0,
            "orientation_variant_reserved": 0,
            "arm_side_variant_attempted": 0,
            "arm_side_variant_width_valid": 0,
            "arm_side_variant_target_capture_valid": 0,
            "arm_side_variant_click_distance_valid": 0,
            "arm_side_variant_visible_collision_valid": 0,
            "arm_side_variant_visible_approach_valid": 0,
            "arm_side_variant_visible_approach_collision_candidates": 0,
            "arm_side_variant_target_solid_collision_rejected": 0,
            "arm_side_variant_background_solid_collision_rejected": 0,
            "arm_side_variant_selected": 0,
            "support_clearance_variant_attempted": 0,
            "support_clearance_variant_width_valid": 0,
            "support_clearance_variant_target_capture_valid": 0,
            "support_clearance_variant_click_distance_valid": 0,
            "support_clearance_variant_visible_collision_valid": 0,
            "support_clearance_variant_visible_approach_valid": 0,
            "support_clearance_variant_visible_approach_collision_candidates": 0,
            "support_clearance_variant_target_solid_collision_rejected": 0,
            "support_clearance_variant_background_solid_collision_rejected": 0,
            "support_clearance_variant_selected": 0,
            "support_clearance_variant_reserved": 0,
            "support_clearance_side_admitted": 0,
            "support_clearance_top_admitted": 0,
        }
        if refine_grasp_centers:
            counts.update({f"centering_variant_{name}": 0 for name in (
                "source_eligible", "source_count", "source_limit_rejected",
                "shift_limit_rejected", "yz_capture_rejected", "attempted",
                "width_valid", "target_capture_valid", "click_distance_valid",
                "visible_collision_valid", "visible_approach_valid",
                "visible_approach_collision_candidates",
                "visible_approach_target_collision_candidates",
                "visible_approach_background_collision_candidates",
                "target_solid_collision_rejected", "background_solid_collision_rejected",
                "selected",
            )})
        if refine_grasp_axial_depth:
            counts.update({f"axial_variant_{name}": 0 for name in (
                "source_count", "attempted", "width_valid", "target_capture_valid",
                "click_distance_valid", "visible_collision_valid", "visible_approach_valid",
                "visible_approach_collision_candidates", "target_solid_collision_rejected",
                "background_solid_collision_rejected", "selected",
            )})
        if refine_grasp_centers:
            counts.update({f"height_variant_{name}": 0 for name in (
                "source_count", "shift_limit_rejected", "attempted", "width_valid",
                "target_capture_valid", "click_distance_valid", "visible_collision_valid",
                "visible_approach_valid", "visible_approach_collision_candidates",
                "target_solid_collision_rejected", "background_solid_collision_rejected", "selected",
            )})
            counts["height_variant_contact_source_count"] = 0
        if proposal_priority is not None and not callable(proposal_priority):
            raise TypeError("proposal_priority must be callable or None.")
        geometry_from_tool = np.linalg.inv(self.solid_geometry.solid_pose(np.eye(4)))
        target_reference = base_from_camera[:3, :3] @ target_camera + base_from_camera[:3, 3]
        geometry_calls = {"surface_normals": int(refine_grasp_centers)}

        def timed_geometry(name, callback, *args, **kwargs):
            # Fixed call-site labels; no extra geometry queries, state cache or
            # ordering changes. These child times must not be added to their
            # enclosing candidate-filtering wall time a second time.
            began = time.perf_counter()
            try:
                return callback(*args, **kwargs)
            finally:
                geometry_calls[name] = geometry_calls.get(name, 0) + 1
                if candidate_timing_ms is not None:
                    key = "candidate_detail_" + name
                    candidate_timing_ms[key] = candidate_timing_ms.get(key, 0.) + (
                        time.perf_counter() - began) * 1000.

        def timed_diversity(*args, **kwargs):
            return timed_geometry("approach_family_diversity", self._distinct_approach_first_indices,
                                  *args, **kwargs)

        def source_priority(item):
            return timed_geometry("source_preference", proposal_priority,
                item.base_from_grasp_geometry @ geometry_from_tool, item.score, target_reference)

        # Only this bounded list retains rejected raw poses for an opt-in retry.
        centering_sources: list[tuple[object, int, float, _RankedGraspCandidate]] = []
        centering_readmissions: list[tuple[int, float, _RankedGraspCandidate | None]] = []
        height_near_misses: list[_RankedGraspCandidate] = []

        def increment(name: str, *, pose_source: str, amount: int = 1) -> None:
            if pose_source == "model_scored":
                key = name
            elif pose_source in {"dmtac_local_orientation_variant_v1", PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1}:
                key = f"orientation_variant_{name}"
            elif pose_source == "dmtac_arm_side_approach_variant_v1":
                key = f"arm_side_variant_{name}"
            elif pose_source == "dmtac_support_clearance_variant_v1":
                key = f"support_clearance_variant_{name}"
            elif pose_source == GRASPGENX_DMTAC_CENTERING_POSE_SOURCE:
                key = f"centering_variant_{name}"
            elif pose_source == GRASPGENX_DMTAC_AXIAL_POSE_SOURCE:
                key = f"axial_variant_{name}"
            elif pose_source == GRASPGENX_DMTAC_HEIGHT_POSE_SOURCE:
                key = f"height_variant_{name}"
            else:
                raise ValueError(f"Unsupported DMTAC pose source: {pose_source}")
            counts[key] = int(counts[key]) + int(amount)

        def prepare_pose(
            *,
            camera_from_grasp_geometry: np.ndarray,
            base_from_grasp_geometry: np.ndarray,
            raw_model_score: float,
            pose_source: str,
            orientation_variant_axis: str | None,
            orientation_variant_angle_deg: float,
            support_clearance_world_z_m: float,
            raw_index: int | None = None,
            centering_shift_m: float = 0.0,
        ) -> _RankedGraspCandidate | None:
            is_variant = pose_source != "model_scored"
            if is_variant:
                increment("attempted", pose_source=pose_source)
            tcp_from_camera = np.linalg.inv(camera_from_grasp_geometry)
            local = (
                target_points_camera @ tcp_from_camera[:3, :3].T
                + tcp_from_camera[:3, 3]
            )
            low, high = timed_geometry("target_quantiles", np.percentile, local, (5.0, 95.0), axis=0)
            target_center = (low + high) * 0.5
            requested_width = (
                float(high[0] - low[0]) + DMTAC_M_GRASPGENX_WIDTH_CLEARANCE_M
            )
            if requested_width > self.max_gripper_width_m:
                return None
            increment("width_valid", pose_source=pose_source)
            downward_cosine = -float(base_from_grasp_geometry[2, 2])
            if not is_variant:
                for name, combine in (
                    ("min_width_valid_downward_approach_cosine", min),
                    ("max_width_valid_downward_approach_cosine", max),
                ):
                    previous = counts[name]
                    counts[name] = (
                        downward_cosine
                        if previous is None
                        else combine(float(previous), downward_cosine)
                    )
                if downward_cosine >= 0.5:
                    counts["width_valid_downward_approach_count"] = (
                        int(counts["width_valid_downward_approach_count"]) + 1
                    )
            width = max(0.005, requested_width)
            # Candidate width is retained as proposal metadata and for the
            # grasp-fit check. It is not an execution input today, so the
            # visible collision proxy must match the open gripper used by the
            # motion planner rather than a narrower candidate-specific proxy.
            visible_collision_opening = self.max_gripper_width_m
            if not is_variant:
                for name, combine in (
                    ("min_visible_collision_opening_m", min),
                    ("max_visible_collision_opening_m", max),
                ):
                    previous = counts[name]
                    counts[name] = (
                        visible_collision_opening
                        if previous is None
                        else combine(float(previous), visible_collision_opening)
                    )
            palm_z = DMTAC_M_GRASPGENX_PALM_Z_M
            fingertip_z = palm_z + DMTAC_M_GRASPGENX_FINGER_LENGTH_M
            # Keep the closing-axis centre close to the jaw mid-plane, while
            # requiring overlap (rather than centroid containment) on the pad
            # height and finger-length axes.  The latter is important for tall
            # objects and partial point clouds: a valid local pinch need not
            # contain the centroid of the whole segmented object.  The previous
            # ``abs(center_x) <= width / 2`` was nearly tautological because the
            # width came from the same span, and admitted strongly one-sided
            # grasps.  The reviewed width-clearance is the bounded centring
            # tolerance instead.
            capture_violation = max(
                0.0,
                abs(float(target_center[0]))
                - DMTAC_M_GRASPGENX_WIDTH_CLEARANCE_M,
                float(low[1]) - DMTAC_M_GRASPGENX_PAD_HEIGHT_M / 2.0,
                -DMTAC_M_GRASPGENX_PAD_HEIGHT_M / 2.0 - float(high[1]),
                palm_z - float(high[2]),
                float(low[2]) - fingertip_z,
            )
            if not is_variant:
                previous_violation = counts["min_target_capture_violation_m"]
                if previous_violation is None or capture_violation < float(
                    previous_violation
                ):
                    counts["min_target_capture_violation_m"] = capture_violation
                    counts["closest_target_center_tcp_x_m"] = float(target_center[0])
                    counts["closest_target_center_tcp_y_m"] = float(target_center[1])
                    counts["closest_target_center_tcp_z_m"] = float(target_center[2])
                downward_violation = counts["min_downward_target_capture_violation_m"]
                if downward_cosine >= 0.5 and (
                    downward_violation is None
                    or capture_violation < float(downward_violation)
                ):
                    counts["min_downward_target_capture_violation_m"] = (
                        capture_violation
                    )
                    counts["closest_downward_target_center_tcp_x_m"] = float(
                        target_center[0]
                    )
                    counts["closest_downward_target_center_tcp_y_m"] = float(
                        target_center[1]
                    )
                    counts["closest_downward_target_center_tcp_z_m"] = float(
                        target_center[2]
                    )
            if capture_violation > 0.0:
                # A missed height band is a refinement source, never an
                # admitted candidate. Keep closing centring and axial overlap
                # unchanged; readmission rechecks all three dimensions.
                if (refine_grasp_centers and not is_variant
                        and abs(float(target_center[0])) <= DMTAC_M_GRASPGENX_WIDTH_CLEARANCE_M
                        and float(high[2]) >= palm_z and float(low[2]) <= fingertip_z):
                    height_support = timed_geometry("pad_height", parallel_jaw_pad_height_support,
                        local, geometry=capture_geometry, opening_m=self.max_gripper_width_m)
                    # Density may prefer a distant body over a nearby closing
                    # patch. Defer shift feasibility until bounded contact
                    # enrichment; this is not candidate admission.
                    parent_score = float(np.clip(raw_model_score, 0., 1.))
                    distance = float(np.linalg.norm(camera_from_grasp_geometry[:3,3]-target_camera))
                    height_near_misses.append(_RankedGraspCandidate(
                        score=parent_score*math.exp(-distance/.060), model_score=parent_score,
                        click_distance_m=distance, gripper_width_m=width,
                        camera_from_grasp_geometry=camera_from_grasp_geometry,
                        base_from_grasp_geometry=base_from_grasp_geometry,
                        capture_height_support_score=height_support.score,
                        capture_height_center_shift_m=height_support.center_shift_m,
                        raw_index=raw_index))
                if (
                    refine_grasp_centers and not is_variant
                    and abs(float(target_center[0])) > DMTAC_M_GRASPGENX_WIDTH_CLEARANCE_M
                ):
                    yz_violation = max(
                        0.0, float(low[1]) - DMTAC_M_GRASPGENX_PAD_HEIGHT_M / 2.0,
                        -DMTAC_M_GRASPGENX_PAD_HEIGHT_M / 2.0 - float(high[1]),
                        palm_z - float(high[2]), float(low[2]) - fingertip_z,
                    )
                    shift = float(target_center[0])
                    if yz_violation > 0.0:
                        increment("yz_capture_rejected", pose_source=GRASPGENX_DMTAC_CENTERING_POSE_SOURCE)
                    elif abs(shift) > GRASPGENX_DMTAC_CENTERING_MAX_SHIFT_M:
                        increment("shift_limit_rejected", pose_source=GRASPGENX_DMTAC_CENTERING_POSE_SOURCE)
                    else:
                        increment("source_eligible", pose_source=GRASPGENX_DMTAC_CENTERING_POSE_SOURCE)
                        parent_score = float(np.clip(raw_model_score, 0.0, 1.0))
                        parent_distance = float(np.linalg.norm(
                            camera_from_grasp_geometry[:3, 3] - target_camera))
                        source = _RankedGraspCandidate(
                            score=parent_score * math.exp(-parent_distance / 0.060),
                            model_score=parent_score, click_distance_m=parent_distance,
                            gripper_width_m=width,
                            camera_from_grasp_geometry=camera_from_grasp_geometry,
                            base_from_grasp_geometry=base_from_grasp_geometry)
                        key = source_priority(source) if proposal_priority is not None else -float(raw_model_score)
                        centering_sources.append((key, raw_index, shift, source))
                        centering_sources.sort(key=lambda row: (row[0], row[1]))
                        # Reserve each observed approach cone before filling
                        # by cost. A dense side family must not evict every
                        # top proposal before physical readmission can run.
                        _, source_indices = timed_diversity(
                            tuple(row[3].base_from_grasp_geometry for row in centering_sources),
                            maximum=GRASPGENX_DMTAC_CENTERING_SOURCE_LIMIT)
                        centering_sources[:] = [centering_sources[i] for i in source_indices]
                return None
            support_count, support_score = timed_geometry("axial_support", parallel_jaw_capture_support,
                local, geometry=capture_geometry,
                opening_m=self.max_gripper_width_m)
            if not support_count:
                counts["target_capture_joint_empty_rejected"] += 1
                return None
            increment("target_capture_valid", pose_source=pose_source)
            if not is_variant and downward_cosine >= 0.5:
                counts["target_capture_downward_approach_count"] = (
                    int(counts["target_capture_downward_approach_count"]) + 1
                )
            centered_camera_from_model_base = (
                camera_from_grasp_geometry @ np.linalg.inv(model_base_from_tcp)
            )
            click_distance = float(
                np.linalg.norm(centered_camera_from_model_base[:3, 3] - target_camera)
            )
            if click_distance > self.max_click_distance_m:
                return None
            increment("click_distance_valid", pose_source=pose_source)
            # These soft preferences cannot rescue a failed capture/click gate.
            # Compute them only for survivors; the raw height-recovery branch
            # above still evaluates its own source before any readmission.
            height_support = timed_geometry("pad_height", parallel_jaw_pad_height_support,
                local, geometry=capture_geometry, opening_m=self.max_gripper_width_m)
            opposed_support = timed_geometry("opposed_support", parallel_jaw_opposed_capture_support,
                local, geometry=capture_geometry, opening_m=self.max_gripper_width_m)
            model_score = float(np.clip(raw_model_score, 0.0, 1.0))
            score = model_score * math.exp(-click_distance / 0.060)
            if is_variant:
                score *= math.exp(
                    -abs(orientation_variant_angle_deg)
                    / GRASPGENX_DMTAC_ORIENTATION_VARIANT_SCORE_DECAY_DEG
                )
            if support_clearance_world_z_m > 0.0:
                score *= math.exp(
                    -support_clearance_world_z_m
                    / GRASPGENX_DMTAC_SUPPORT_CLEARANCE_SCORE_DECAY_M
                )
            if pose_source == GRASPGENX_DMTAC_CENTERING_POSE_SOURCE:
                score *= math.exp(-abs(centering_shift_m) / GRASPGENX_DMTAC_CENTERING_SCORE_DECAY_M)
            return _RankedGraspCandidate(
                score=score,
                model_score=model_score,
                click_distance_m=click_distance,
                gripper_width_m=width,
                camera_from_grasp_geometry=camera_from_grasp_geometry,
                base_from_grasp_geometry=base_from_grasp_geometry,
                pose_source=pose_source,
                orientation_variant_axis=orientation_variant_axis,
                orientation_variant_angle_deg=orientation_variant_angle_deg,
                support_clearance_world_z_m=support_clearance_world_z_m,
                capture_support_score=support_score,
                opposed_capture_support_score=opposed_support.score,
                capture_height_support_score=height_support.score,
                capture_height_center_shift_m=height_support.center_shift_m,
                raw_index=raw_index,
            )

        terminal_collision_evidence_by_candidate: dict[int, tuple[int, int]] = {}
        surface_observation_replacements: list[tuple[_RankedGraspCandidate, _RankedGraspCandidate]] = []

        def with_surface_observation(item):
            model_from_camera = np.linalg.inv(item.camera_from_grasp_geometry)
            surface = timed_geometry("surface_alignment", parallel_jaw_surface_alignment,
                target_points_camera @ model_from_camera[:3, :3].T + model_from_camera[:3, 3],
                surface_normals @ model_from_camera[:3, :3].T, surface_confidence,
                geometry=capture_geometry, opening_m=self.max_gripper_width_m)
            return replace(item, surface_alignment_score=surface.score,
                surface_alignment_observed_sides=surface.observed_sides,
                surface_opposition_score=surface.opposition_score,
                contact_height_support_score=surface.pad_height_score,
                contact_height_center_shift_m=surface.center_shift_m,
                contact_height_evidence_weight=surface.height_evidence_weight)

        def admit_visible_scene(
            item: _RankedGraspCandidate,
        ) -> _RankedGraspCandidate | None:
            is_variant = item.pose_source != "model_scored"
            downward_cosine = -float(item.base_from_grasp_geometry[2, 2])
            target_hits, target_contact_points, background_hits = (
                timed_geometry("terminal_collision", terminal_scene.collision_counts_with_terminal_target_contact,
                    item.camera_from_grasp_geometry,
                    opening_m=self.max_gripper_width_m,
                    contact_depth_m=(DMTAC_M_GRASPGENX_TERMINAL_TARGET_CONTACT_DEPTH_M),
                    **({"proxy_padding_m": self.background_proxy_padding_m}
                       if self.background_proxy_padding_m else {}),
                )
            )
            terminal_collision_evidence_by_candidate[id(item)] = (
                int(target_hits),
                int(background_hits),
            )
            if not is_variant:
                counts["terminal_target_contact_points"] = (
                    int(counts["terminal_target_contact_points"])
                    + target_contact_points
                )
            if target_hits or background_hits:
                increment(
                    "target_solid_collision_rejected",
                    pose_source=item.pose_source,
                    amount=int(bool(target_hits)),
                )
                increment(
                    "background_solid_collision_rejected",
                    pose_source=item.pose_source,
                    amount=int(bool(background_hits)),
                )
                if not is_variant and background_hits:
                    counts["terminal_background_collision_rejected"] = (
                        int(counts["terminal_background_collision_rejected"]) + 1
                    )
                return None
            increment("visible_collision_valid", pose_source=item.pose_source)
            approach_target_hits, approach_background_hits = (
                timed_geometry("approach_collision", terminal_scene.approach_collision_counts,
                    item.camera_from_grasp_geometry,
                    opening_m=self.max_gripper_width_m,
                    distance_m=0.060,
                    allow_terminal_target_contact=True,
                    terminal_target_contact_depth_m=(
                        DMTAC_M_GRASPGENX_TERMINAL_TARGET_CONTACT_DEPTH_M
                    ),
                )
            )
            if approach_target_hits or approach_background_hits:
                increment(
                    "visible_approach_collision_candidates",
                    pose_source=item.pose_source,
                )
                if not is_variant:
                    counts["visible_approach_target_collision_candidates"] = int(
                        counts["visible_approach_target_collision_candidates"]
                    ) + int(bool(approach_target_hits))
                    counts["visible_approach_background_collision_candidates"] = int(
                        counts["visible_approach_background_collision_candidates"]
                    ) + int(bool(approach_background_hits))
                elif item.pose_source == GRASPGENX_DMTAC_CENTERING_POSE_SOURCE:
                    increment("visible_approach_target_collision_candidates", pose_source=item.pose_source,
                              amount=int(bool(approach_target_hits)))
                    increment("visible_approach_background_collision_candidates", pose_source=item.pose_source,
                              amount=int(bool(approach_background_hits)))
            else:
                increment("visible_approach_valid", pose_source=item.pose_source)
            if not is_variant:
                for name, combine in (
                    ("min_downward_approach_cosine", min),
                    ("max_downward_approach_cosine", max),
                ):
                    previous = counts[name]
                    counts[name] = (
                        downward_cosine
                        if previous is None
                        else combine(float(previous), downward_cosine)
                    )
                if downward_cosine >= 0.5:
                    counts["downward_approach_count"] = (
                        int(counts["downward_approach_count"]) + 1
                    )
            if surface_normals is not None:
                # Contact quality cannot rescue a terminal solid collision.
                # Evaluate it only after physical readmission; dense scenes
                # otherwise spend most of this work on rejected source poses.
                original = item
                item = with_surface_observation(item)
                # Keep both objects alive until pool indexing. Refinement
                # sources may already reference the immutable pre-observation
                # candidate; enriching its evidence must not lose lineage.
                surface_observation_replacements.append((original, item))
                terminal_collision_evidence_by_candidate[id(item)] = (int(target_hits), int(background_hits))
            return item

        for raw_index, (base_from_model_base, raw_score) in enumerate(
            zip(grasps, scores, strict=True)
        ):
            counts["detected"] = int(counts["detected"]) + 1
            try:
                base_from_model_base = _normalize_graspgenx_rigid_matrix(
                    base_from_model_base,
                    field_name="GraspGenX base_from_model_base",
                )
            except ValueError:
                continue
            base_from_tcp = base_from_model_base @ model_base_from_tcp
            camera_from_tcp = camera_from_base @ base_from_tcp
            # Preserve both translation and orientation of each scored pose.
            # Finger exchange preserves the nominal jaw solids, not the whole
            # mounted robot or its eccentric controller TCP and IK feasibility.
            camera_from_tcp = _strict_rigid_matrix(
                camera_from_tcp,
                field_name="GraspGenX camera_from_tcp candidate",
            )
            base_from_tcp = _strict_rigid_matrix(
                base_from_tcp,
                field_name="GraspGenX base_from_tcp candidate",
            )
            prepared = prepare_pose(
                camera_from_grasp_geometry=camera_from_tcp,
                base_from_grasp_geometry=base_from_tcp,
                raw_model_score=float(raw_score),
                pose_source="model_scored",
                orientation_variant_axis=None,
                orientation_variant_angle_deg=0.0,
                support_clearance_world_z_m=0.0,
                raw_index=raw_index,
            )
            if prepared is not None:
                ranked.append(prepared)

        ranked.sort(key=lambda item: (-item.score, item.click_distance_m))
        model_candidates = timed_geometry("deduplication", _deduplicate_ranked_poses, ranked)

        if proposal_priority is not None:
            # A direct planner chooses refinement sources by arm-aware cheap
            # preferences, not only model scores. This does not admit a pose.
            model_candidates.sort(key=source_priority)

        target_points_base = (
            target_points_camera @ base_from_camera[:3, :3].T
            + base_from_camera[:3, 3]
        )
        target_base_low, target_base_high = np.percentile(
            target_points_base,
            (5.0, 95.0),
            axis=0,
        )
        target_center_base = (target_base_low + target_base_high) * 0.5

        def prepare_orientation_variants(
            source: _RankedGraspCandidate,
            *, axis_angles=None,
        ) -> list[_RankedGraspCandidate]:
            variants: list[_RankedGraspCandidate] = []
            source_grasp_from_camera = np.linalg.inv(
                source.camera_from_grasp_geometry
            )
            source_target_local = (
                target_points_camera @ source_grasp_from_camera[:3, :3].T
                + source_grasp_from_camera[:3, 3]
            )
            source_target_low, source_target_high = np.percentile(
                source_target_local,
                (5.0, 95.0),
                axis=0,
            )
            source_target_center_local = (
                source_target_low + source_target_high
            ) * 0.5
            source_target_center_camera = (
                source.camera_from_grasp_geometry[:3, :3]
                @ source_target_center_local
                + source.camera_from_grasp_geometry[:3, 3]
            )
            if axis_angles is None:
                axis_angles = self.orientation_variants_local_axis_degrees
            for axis, angle_deg in axis_angles:
                variant_camera_from_grasp_geometry = (
                    source.camera_from_grasp_geometry
                    @ _local_axis_rotation(axis, angle_deg)
                )
                # Rotate around the observed target center, not around the TCP.
                # Keeping the TCP translation fixed would move the object in
                # gripper-local Y/Z and turn an otherwise useful wrist tilt
                # into a target/palm collision. Preserve the target's local
                # capture position while changing only the approach plane.
                variant_camera_from_grasp_geometry[:3, 3] = (
                    source_target_center_camera
                    - variant_camera_from_grasp_geometry[:3, :3]
                    @ source_target_center_local
                )
                variant_camera_from_grasp_geometry = _strict_rigid_matrix(
                    variant_camera_from_grasp_geometry,
                    field_name="DMTAC camera orientation variant",
                )
                variant = prepare_pose(
                    camera_from_grasp_geometry=variant_camera_from_grasp_geometry,
                    base_from_grasp_geometry=_strict_rigid_matrix(
                        base_from_camera @ variant_camera_from_grasp_geometry,
                        field_name="DMTAC base orientation variant",
                    ),
                    raw_model_score=source.model_score,
                    raw_index=source.raw_index,
                    pose_source=(PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1
                        if axis == "z" and abs(angle_deg) == 180.0
                        else "dmtac_local_orientation_variant_v1"),
                    orientation_variant_axis=axis,
                    orientation_variant_angle_deg=angle_deg,
                    support_clearance_world_z_m=0.0,
                )
                if variant is not None:
                    if axis == "z" and abs(angle_deg) == 180.0:
                        variant = replace(variant, symmetry_parent=source,
                            score=source.score * math.exp(-abs(angle_deg)
                                / GRASPGENX_DMTAC_ORIENTATION_VARIANT_SCORE_DECAY_DEG))
                    variants.append(variant)
            return variants

        def prepare_arm_side_variant(
            source: _RankedGraspCandidate,
            origin: tuple[float, float, float],
        ) -> list[_RankedGraspCandidate]:
            source_approach = np.asarray(
                source.base_from_grasp_geometry[:3, 2], dtype=np.float64
            )
            desired_horizontal = (
                target_center_base[:2]
                - np.asarray(origin[:2])
            )
            desired_norm = float(np.linalg.norm(desired_horizontal))
            if desired_norm < 1e-6:
                return []
            desired_unit = desired_horizontal / desired_norm
            top_approach = (
                -float(source_approach[2])
                >= GRASPGENX_DMTAC_TOP_APPROACH_COSINE
            )
            source_grasp_from_camera = np.linalg.inv(
                source.camera_from_grasp_geometry
            )
            source_target_local = (
                target_points_camera @ source_grasp_from_camera[:3, :3].T
                + source_grasp_from_camera[:3, 3]
            )
            source_target_low, source_target_high = np.percentile(
                source_target_local,
                (5.0, 95.0),
                axis=0,
            )
            source_target_center_local = (
                source_target_low + source_target_high
            ) * 0.5
            if not top_approach:
                source_horizontal = source_approach[:2]
                source_norm = float(np.linalg.norm(source_horizontal))
                if source_norm < 1e-6:
                    return []
                source_unit = source_horizontal / source_norm
                side_angle_deg = math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                np.dot(source_unit, desired_unit), -1.0, 1.0
                            )
                        )
                    )
                )
                if side_angle_deg > (
                    GRASPGENX_DMTAC_ARM_SIDE_VARIANT_MAX_ROTATION_DEG
                ):
                    return []
                variants: list[_RankedGraspCandidate] = []
                # A horizontal closing axis and vertical pad-height axis avoid
                # inheriting a source roll that lowers one long DMTAC finger
                # into the support surface. A bounded horizontal cone avoids
                # collapsing every proposal to one ray while preserving the
                # authorized arm-side prior. Both 180-degree-equivalent wrist
                # branches are retained because they have different RM65 IK.
                for yaw_offset_deg in (
                    GRASPGENX_DMTAC_ARM_SIDE_APPROACH_YAW_OFFSETS_DEG
                ):
                    yaw_offset_rad = math.radians(yaw_offset_deg)
                    c, s = math.cos(yaw_offset_rad), math.sin(yaw_offset_rad)
                    approach_horizontal = np.asarray(
                        (
                            c * desired_unit[0] - s * desired_unit[1],
                            s * desired_unit[0] + c * desired_unit[1],
                        ),
                        dtype=np.float64,
                    )
                    approach_axis = np.asarray(
                        (
                            approach_horizontal[0],
                            approach_horizontal[1],
                            0.0,
                        ),
                        dtype=np.float64,
                    )
                    candidate_rotation_deg = math.degrees(
                        math.acos(
                            float(
                                np.clip(
                                    np.dot(source_unit, approach_horizontal),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                    if candidate_rotation_deg > (
                        GRASPGENX_DMTAC_ARM_SIDE_VARIANT_MAX_ROTATION_DEG
                    ):
                        continue
                    for vertical_sign in (1.0, -1.0):
                        height_axis = np.asarray(
                            (0.0, 0.0, vertical_sign), dtype=np.float64
                        )
                        closing_axis = np.cross(height_axis, approach_axis)
                        closing_axis /= np.linalg.norm(closing_axis)
                        rotation = np.column_stack(
                            (closing_axis, height_axis, approach_axis)
                        )
                        target_center_local = np.asarray(
                            (0.0, 0.0, source_target_center_local[2]),
                            dtype=np.float64,
                        )
                        variant_base_from_grasp_geometry = np.eye(
                            4, dtype=np.float64
                        )
                        variant_base_from_grasp_geometry[:3, :3] = rotation
                        variant_base_from_grasp_geometry[:3, 3] = (
                            target_center_base - rotation @ target_center_local
                        )
                        variant_base_from_grasp_geometry = _strict_rigid_matrix(
                            variant_base_from_grasp_geometry,
                            field_name="DMTAC canonical arm-side base variant",
                        )
                        variant_camera_from_grasp_geometry = _strict_rigid_matrix(
                            camera_from_base @ variant_base_from_grasp_geometry,
                            field_name="DMTAC canonical arm-side camera variant",
                        )
                        variant = prepare_pose(
                            camera_from_grasp_geometry=(
                                variant_camera_from_grasp_geometry
                            ),
                            base_from_grasp_geometry=(
                                variant_base_from_grasp_geometry
                            ),
                            raw_model_score=source.model_score,
                            raw_index=source.raw_index,
                            pose_source="dmtac_arm_side_approach_variant_v1",
                            orientation_variant_axis=None,
                            orientation_variant_angle_deg=(
                                yaw_offset_deg
                            ),
                            support_clearance_world_z_m=0.0,
                        )
                        if variant is not None:
                            variants.append(variant)
                return variants
            source_horizontal = (
                source.base_from_grasp_geometry[:2, 0]
                if top_approach
                else source_approach[:2]
            )
            source_norm = float(np.linalg.norm(source_horizontal))
            if source_norm < 1e-6:
                return []
            source_unit = source_horizontal / source_norm
            desired_directions = [desired_unit]
            if top_approach:
                desired_directions.append(
                    np.asarray((-desired_unit[1], desired_unit[0]))
                )
            variants: list[_RankedGraspCandidate] = []
            for desired_direction in desired_directions:
                rotation_rad = math.atan2(
                    float(
                        source_unit[0] * desired_direction[1]
                        - source_unit[1] * desired_direction[0]
                    ),
                    float(
                        np.clip(np.dot(source_unit, desired_direction), -1.0, 1.0)
                    ),
                )
                rotation_deg = math.degrees(rotation_rad)
                if top_approach:
                    while rotation_deg > 90.0:
                        rotation_deg -= 180.0
                    while rotation_deg < -90.0:
                        rotation_deg += 180.0
                if abs(rotation_deg) < 1e-6 or abs(rotation_deg) > (
                    GRASPGENX_DMTAC_ARM_SIDE_VARIANT_MAX_ROTATION_DEG
                ):
                    continue
                base_rotation = _local_axis_rotation("z", rotation_deg)
                variant_base_from_grasp_geometry = (
                    source.base_from_grasp_geometry.copy()
                )
                variant_base_from_grasp_geometry[:3, :3] = (
                    base_rotation[:3, :3]
                    @ source.base_from_grasp_geometry[:3, :3]
                )
                variant_base_from_grasp_geometry[:3, 3] = (
                    target_center_base
                    + base_rotation[:3, :3]
                    @ (
                        source.base_from_grasp_geometry[:3, 3]
                        - target_center_base
                    )
                )
                variant_base_from_grasp_geometry = _strict_rigid_matrix(
                    variant_base_from_grasp_geometry,
                    field_name="DMTAC arm-side base variant",
                )
                variant_camera_from_grasp_geometry = _strict_rigid_matrix(
                    camera_from_base @ variant_base_from_grasp_geometry,
                    field_name="DMTAC arm-side camera variant",
                )
                variant = prepare_pose(
                    camera_from_grasp_geometry=variant_camera_from_grasp_geometry,
                    base_from_grasp_geometry=variant_base_from_grasp_geometry,
                    raw_model_score=source.model_score,
                    raw_index=source.raw_index,
                    pose_source="dmtac_arm_side_approach_variant_v1",
                    orientation_variant_axis=None,
                    orientation_variant_angle_deg=rotation_deg,
                    support_clearance_world_z_m=0.0,
                )
                if variant is not None:
                    variants.append(variant)
            return variants

        _representatives, variant_source_indices = (
            timed_diversity(
                tuple(
                    item.camera_from_grasp_geometry for item in model_candidates
                ),
                maximum=min(
                    self.orientation_variant_source_limit,
                    len(model_candidates),
                ),
            )
        )
        prepared_candidates = list(model_candidates)
        if refine_grasp_centers:
            counts["centering_variant_source_count"] = len(centering_sources)
            counts["centering_variant_source_limit_rejected"] = (
                int(counts["centering_variant_source_eligible"]) - len(centering_sources))
            for _key, parent_index, shift, source in centering_sources:
                # Translate a separate model-frame pose; never mutate its scored parent.
                camera_pose = source.camera_from_grasp_geometry.copy()
                camera_pose[:3, 3] += camera_pose[:3, 0] * shift
                variant = prepare_pose(
                    camera_from_grasp_geometry=_strict_rigid_matrix(
                        camera_pose, field_name="DMTAC centered camera variant"),
                    base_from_grasp_geometry=_strict_rigid_matrix(
                        base_from_camera @ camera_pose, field_name="DMTAC centered base variant"),
                    raw_model_score=source.model_score,
                    pose_source=GRASPGENX_DMTAC_CENTERING_POSE_SOURCE,
                    orientation_variant_axis=None, orientation_variant_angle_deg=0.0,
                    support_clearance_world_z_m=0.0, centering_shift_m=shift, raw_index=parent_index)
                centering_readmissions.append((parent_index, shift, variant))
                if variant is not None:
                    prepared_candidates.append(variant)
        for source_index in variant_source_indices:
            source = model_candidates[source_index]
            prepared_candidates.extend(prepare_orientation_variants(source))
        _arm_representatives, distinct_arm_source_indices = (
            timed_diversity(
                tuple(
                    item.camera_from_grasp_geometry for item in model_candidates
                ),
                maximum=min(
                    GRASPGENX_DMTAC_ARM_SIDE_VARIANT_SOURCE_LIMIT,
                    len(model_candidates),
                ),
            )
        )
        arm_source_indices = list(
            range(
                min(
                    GRASPGENX_DMTAC_ARM_SIDE_VARIANT_SOURCE_LIMIT // 2,
                    len(model_candidates),
                )
            )
        )
        for source_index in distinct_arm_source_indices:
            if source_index not in arm_source_indices:
                arm_source_indices.append(source_index)
            if len(arm_source_indices) >= GRASPGENX_DMTAC_ARM_SIDE_VARIANT_SOURCE_LIMIT:
                break
        for source_index in arm_source_indices:
            for origin in self.preferred_approach_origins_base_m:
                prepared_candidates.extend(
                    prepare_arm_side_variant(model_candidates[source_index], origin)
                )

        ranked = []
        admitted_pose_variants: list[np.ndarray] = []
        blocked_model_candidates: list[_RankedGraspCandidate] = []
        blocked_arm_side_candidates: list[_RankedGraspCandidate] = []
        admitted_model_candidates: list[_RankedGraspCandidate] = []
        for item in prepared_candidates:
            admitted = admit_visible_scene(item)
            if admitted is not None and not _pose_is_duplicate(
                admitted.camera_from_grasp_geometry, admitted_pose_variants):
                ranked.append(admitted)
                admitted_pose_variants.append(
                    admitted.camera_from_grasp_geometry
                )
            if item.pose_source == "model_scored":
                if admitted is None:
                    blocked_model_candidates.append(item)
                else:
                    admitted_model_candidates.append(admitted)
            elif (
                item.pose_source == "dmtac_arm_side_approach_variant_v1"
                and admitted is None
            ):
                blocked_arm_side_candidates.append(item)

        # Pre-collision source selection preserves raw approach diversity, but
        # the few model poses that survive a dense observed scene can be low in
        # the score order and therefore absent from that source set. If the
        # shortlist remains short, derive bounded wrist variants from those
        # physically admitted poses and run the complete admission check again.
        if proposal_priority is not None or len(ranked) < internal_candidate_target:
            if proposal_priority is not None:
                sources = sorted(admitted_model_candidates, key=source_priority)
                source_order = diverse_grasp_search_order(
                    tuple(item.base_from_grasp_geometry for item in sources),
                    tuple(range(len(sources))), policy=FirstGraspPathPolicy())
                sources = [sources[index] for index in source_order[:self.orientation_variant_source_limit]]
            else:
                sources = admitted_model_candidates
            for source in sources:
                for variant in prepare_orientation_variants(source):
                    if _pose_is_duplicate(variant.camera_from_grasp_geometry, admitted_pose_variants):
                        continue
                    admitted = admit_visible_scene(variant)
                    if admitted is None:
                        continue
                    ranked.append(admitted)
                    admitted_pose_variants.append(
                        admitted.camera_from_grasp_geometry
                    )
                    if proposal_priority is None and len(ranked) >= internal_candidate_target:
                        break
                if proposal_priority is None and len(ranked) >= internal_candidate_target:
                    break

        # The dominant false bottleneck in saved table scenes is a physically
        # plausible grasp whose long DMTAC finger/palm proxy intersects the
        # observed support surface at its model-proposed depth.  Search a
        # bounded set of upward world-Z refinements for distinct model poses.
        # Try collision-blocked poses first.  With an embodiment selector,
        # visible-scene abundance is not evidence of whole-arm feasibility, so
        # retain a bounded clearance family even when the perception shortlist
        # is already full.  Without one, preserve the generic underfill-only
        # behavior. Every refined pose is sent back through target capture,
        # click distance, and the exact same zero-background terminal collision
        # check; this never turns collision points into a tolerance.
        support_candidates: list[_RankedGraspCandidate] = []
        support_candidate_poses = [
            item.camera_from_grasp_geometry for item in ranked
        ]
        support_target_count = min(
            GRASPGENX_DMTAC_SUPPORT_CLEARANCE_RETURN_LIMIT,
            (
                GRASPGENX_DMTAC_SUPPORT_CLEARANCE_RETURN_LIMIT
                if for_embodiment
                else max(0, internal_candidate_target - len(ranked))
            ),
        )
        if proposal_priority is not None:
            # Direct first-path search has no five-candidate publication page.
            # Keep a bounded internal set across depth/orientation families;
            # an abundant blocked family must not consume every refinement.
            support_target_count = GRASPGENX_DMTAC_SUPPORT_CLEARANCE_RETURN_LIMIT * 8
        def append_support_candidates(
            sources: list[_RankedGraspCandidate],
            *,
            limit: int,
        ) -> None:
            if not sources or len(support_candidates) >= limit:
                return
            sources.sort(
                key=lambda item: (
                    source_priority(item) if proposal_priority is not None else (),
                    sum(terminal_collision_evidence_by_candidate[id(item)]),
                    terminal_collision_evidence_by_candidate[id(item)][1],
                    terminal_collision_evidence_by_candidate[id(item)][0],
                    -item.score,
                )
            )
            _representatives, source_indices = (
                timed_diversity(
                    tuple(item.camera_from_grasp_geometry for item in sources),
                    maximum=min(
                        GRASPGENX_DMTAC_SUPPORT_CLEARANCE_SOURCE_LIMIT,
                        len(sources),
                    ),
                )
            )
            if proposal_priority is not None:
                source_indices = diverse_grasp_search_order(
                    tuple(item.base_from_grasp_geometry for item in sources),
                    tuple(range(len(sources))), policy=FirstGraspPathPolicy())[
                        :GRASPGENX_DMTAC_SUPPORT_CLEARANCE_SOURCE_LIMIT]
            # Reserve height diversity before filling from the remaining
            # sources. Otherwise four nominally valid +5 mm poses consume a
            # side-family budget and hide its +10/+15 mm executable variants.
            # Every deferred pose has passed the same physical readmission.
            slots = limit - len(support_candidates)
            per_height = (max(1, slots // len(GRASPGENX_DMTAC_SUPPORT_CLEARANCE_WORLD_Z_M))
                          if proposal_priority is not None else limit)
            deferred_support: list[_RankedGraspCandidate] = []
            for clearance_m in GRASPGENX_DMTAC_SUPPORT_CLEARANCE_WORLD_Z_M:
                admitted_at_height = 0
                for source_index in source_indices:
                    source = sources[source_index]
                    variant_base_from_grasp_geometry = (
                        source.base_from_grasp_geometry.copy()
                    )
                    variant_base_from_grasp_geometry[2, 3] += clearance_m
                    variant_base_from_grasp_geometry = _strict_rigid_matrix(
                        variant_base_from_grasp_geometry,
                        field_name="DMTAC support-clearance base variant",
                    )
                    variant_camera_from_grasp_geometry = _strict_rigid_matrix(
                        camera_from_base @ variant_base_from_grasp_geometry,
                        field_name="DMTAC support-clearance camera variant",
                    )
                    variant = prepare_pose(
                        camera_from_grasp_geometry=(
                            variant_camera_from_grasp_geometry
                        ),
                        base_from_grasp_geometry=(
                            variant_base_from_grasp_geometry
                        ),
                        raw_model_score=source.model_score,
                        raw_index=source.raw_index,
                        pose_source="dmtac_support_clearance_variant_v1",
                        orientation_variant_axis=None,
                        orientation_variant_angle_deg=0.0,
                        support_clearance_world_z_m=clearance_m,
                    )
                    if variant is None:
                        continue
                    admitted = admit_visible_scene(variant)
                    if admitted is None or _pose_is_duplicate(
                        admitted.camera_from_grasp_geometry, support_candidate_poses):
                        continue
                    support_candidate_poses.append(
                        admitted.camera_from_grasp_geometry
                    )
                    if admitted_at_height >= per_height:
                        deferred_support.append(admitted)
                        if len(deferred_support) >= slots:
                            break
                        continue
                    admitted_at_height += 1
                    support_candidates.append(admitted)
                    group = (
                        "top" if -float(admitted.base_from_grasp_geometry[2, 2])
                        >= GRASPGENX_DMTAC_TOP_APPROACH_COSINE else "side"
                    )
                    counts[f"support_clearance_{group}_admitted"] = (
                        int(counts[f"support_clearance_{group}_admitted"]) + 1
                    )
                    if len(support_candidates) >= limit:
                        break
                    if admitted_at_height >= per_height and len(deferred_support) >= slots:
                        break
                if len(support_candidates) >= limit:
                    break
            for admitted in deferred_support:
                if len(support_candidates) >= limit:
                    break
                support_candidates.append(admitted)
                group = (
                    "top" if -float(admitted.base_from_grasp_geometry[2, 2])
                    >= GRASPGENX_DMTAC_TOP_APPROACH_COSINE else "side"
                )
                counts[f"support_clearance_{group}_admitted"] = int(
                    counts[f"support_clearance_{group}_admitted"]) + 1
        if proposal_priority is not None:
            append_support_candidates(
                blocked_arm_side_candidates + blocked_model_candidates + admitted_model_candidates,
                limit=support_target_count)
        elif support_target_count:
            # The closest-to-collision ordering otherwise fills all eight
            # clearance slots with near-identical top grasps. Give side
            # approaches their own bounded search before filling the remaining
            # slots; every pose still passes the exact target and background
            # solid checks above.
            blocked_side_candidates = [
                item for item in blocked_arm_side_candidates
                if -float(item.base_from_grasp_geometry[2, 2])
                < GRASPGENX_DMTAC_TOP_APPROACH_COSINE
            ]
            blocked_top_candidates = [
                item for item in blocked_arm_side_candidates
                if -float(item.base_from_grasp_geometry[2, 2])
                >= GRASPGENX_DMTAC_TOP_APPROACH_COSINE
            ]
            append_support_candidates(
                blocked_side_candidates,
                limit=min(4, support_target_count),
            )
            append_support_candidates(
                blocked_top_candidates,
                limit=support_target_count,
            )
        if len(support_candidates) < support_target_count:
            append_support_candidates(
                blocked_model_candidates, limit=support_target_count
            )
        if len(support_candidates) < support_target_count:
            append_support_candidates(
                admitted_model_candidates, limit=support_target_count
            )
        ranked.extend(support_candidates)

        if proposal_priority is not None:
            sources = sorted(support_candidates, key=source_priority)
            source_order = diverse_grasp_search_order(
                tuple(item.base_from_grasp_geometry for item in sources),
                tuple(range(len(sources))), policy=FirstGraspPathPolicy())
            for index in source_order[:self.orientation_variant_source_limit]:
                for variant in prepare_orientation_variants(sources[index]):
                    if _pose_is_duplicate(variant.camera_from_grasp_geometry, admitted_pose_variants):
                        continue
                    admitted = admit_visible_scene(variant)
                    if admitted is not None:
                        ranked.append(admitted)
                        admitted_pose_variants.append(admitted.camera_from_grasp_geometry)

        if for_embodiment and self.enable_finger_exchange:
            # A valid tilted/support-refined proposal also needs its own finger
            # exchange. Generating it only from raw poses misses that branch.
            sources = sorted(ranked, key=lambda item: (-item.score, item.click_distance_m))
            _, source_indices = timed_diversity(
                tuple(item.camera_from_grasp_geometry for item in sources),
                maximum=min(internal_candidate_target, len(sources)))
            for index in source_indices:
                for variant in prepare_orientation_variants(sources[index], axis_angles=(("z", 180.),)):
                    admitted = admit_visible_scene(variant)
                    if admitted is not None:
                        ranked.append(admitted)

        ranked.sort(key=lambda item: (-item.score, item.click_distance_m))
        deduplicated = timed_geometry("deduplication", _deduplicate_ranked_poses, ranked)

        height_readmissions = []
        if refine_grasp_centers:
            policy = FirstGraspPathPolicy()

            def height_proposal(item):
                # Use observed closing faces even when the dense top band
                # misses. One bounded pass must not stop at density recovery.
                if grasp_contact_height_observed(item, policy=policy):
                    return item.contact_height_center_shift_m, "observed_contact_band"
                if item.capture_height_support_score < policy.minimum_capture_height_support_score:
                    return item.capture_height_center_shift_m, "observed_density_band"
                return item.contact_height_center_shift_m, "observed_contact_band"

            shallow_height = [item for item in (*deduplicated, *height_near_misses)
                              if (item.capture_height_support_score < policy.minimum_capture_height_support_score
                                  or item.contact_height_support_score < 1.-1e-9)]
            height_miss_ids = {id(item) for item in height_near_misses}
            sources = [item for item in shallow_height
                       if id(item) in height_miss_ids or _pad_height_shifts(height_proposal(item)[0])]
            counts["height_variant_shift_limit_rejected"] += len(shallow_height) - len(sources)
            sources.sort(key=lambda item: (
                source_priority(item) if proposal_priority is not None else -item.score,
                abs(height_proposal(item)[0])))
            order = diverse_grasp_search_order(
                tuple(item.base_from_grasp_geometry for item in sources),
                tuple(range(len(sources))), policy=policy)
            for index in order[:GRASPGENX_DMTAC_HEIGHT_SOURCE_LIMIT]:
                source = sources[index]
                counts["height_variant_source_count"] += 1
                if id(source) in height_miss_ids and surface_normals is not None:
                    # This source failed overlap, so it never reached terminal
                    # admission/normal enrichment. Evaluate only the bounded
                    # selected sources; their children must still pass every gate.
                    source = with_surface_observation(source)
                center_shift, shift_basis = height_proposal(source)
                deltas = _pad_height_shifts(center_shift)
                if not deltas:
                    counts["height_variant_shift_limit_rejected"] += 1
                    continue
                if shift_basis == "observed_contact_band":
                    counts["height_variant_contact_source_count"] += 1
                for delta in deltas:
                    if proposal_priority is not None:
                        source_priority(source)
                    pose = source.camera_from_grasp_geometry.copy()
                    pose[:3,3] += pose[:3,1]*delta
                    child = prepare_pose(
                        camera_from_grasp_geometry=pose,
                        base_from_grasp_geometry=base_from_camera@pose,
                        raw_model_score=source.model_score, raw_index=source.raw_index,
                        pose_source=GRASPGENX_DMTAC_HEIGHT_POSE_SOURCE,
                        orientation_variant_axis=None, orientation_variant_angle_deg=0.,
                        support_clearance_world_z_m=0.)
                    if child is not None:
                        child = replace(child, height_parent=source, height_shift_m=delta,
                            height_shift_basis=shift_basis,
                            score=source.score*math.exp((source.click_distance_m-child.click_distance_m)/.060))
                        child = admit_visible_scene(child)
                    if (child is not None
                            and grasp_capture_height_acceptable(child, policy=policy)
                            and not _pose_is_duplicate(child.camera_from_grasp_geometry,
                                [item.camera_from_grasp_geometry for item in deduplicated])):
                        deduplicated.append(child)
                        height_readmissions.append((source, child))

        contact_orientation_readmissions = []
        if refine_grasp_centers and proposal_priority is not None:
            # Rotation before height correction and rotation after correction
            # have different contact centres. Reuse the existing finite local
            # orientations on at most its normal source allowance, never an
            # unbounded Cartesian product or an unvalidated executable offset.
            sources = sorted((child for _parent, child in height_readmissions
                if grasp_contact_height_observed(child, policy=policy)), key=source_priority)
            order = diverse_grasp_search_order(
                tuple(item.base_from_grasp_geometry for item in sources),
                tuple(range(len(sources))), policy=policy)
            for index in order[:min(self.orientation_variant_source_limit,
                    GRASPGENX_DMTAC_ORIENTATION_VARIANT_SOURCE_LIMIT)]:
                for variant in prepare_orientation_variants(sources[index]):
                    if _pose_is_duplicate(variant.camera_from_grasp_geometry,
                            [item.camera_from_grasp_geometry for item in deduplicated]):
                        continue
                    child = admit_visible_scene(variant)
                    if child is not None and grasp_capture_height_acceptable(child, policy=policy):
                        deduplicated.append(child)
                        contact_orientation_readmissions.append((sources[index], child))

        axial_readmissions = []
        if refine_grasp_axial_depth:
            # Only refine already admitted, distinct parent poses. Keep each
            # original intact; a child is never a newly neurally scored pose.
            axial_policy = FirstGraspPathPolicy()
            # Deep grasps need no axial proposal and must not consume the
            # finite shallow-source allowance before its eligibility filter.
            shallow = [item for item in deduplicated
                if item.capture_support_score < GRASPGENX_DMTAC_AXIAL_SHALLOW_SUPPORT]
            sources = sorted(shallow, key=lambda item: (
                source_priority(item), grasp_capture_support_cost(item, policy=axial_policy)
            )) if proposal_priority else shallow
            source_order = diverse_grasp_search_order(
                tuple(item.base_from_grasp_geometry @ geometry_from_tool for item in sources),
                tuple(range(len(sources))), policy=axial_policy)
            for index in source_order[:GRASPGENX_DMTAC_AXIAL_SOURCE_LIMIT]:
                source = sources[index]
                counts["axial_variant_source_count"] += 1
                admitted_children = []
                for shift in GRASPGENX_DMTAC_AXIAL_SHIFTS_M:
                    # The existing callback also enforces cancellation/deadline.
                    if proposal_priority is not None:
                        source_priority(source)
                    camera_pose = source.camera_from_grasp_geometry.copy()
                    camera_pose[:3, 3] += camera_pose[:3, 2] * shift
                    child = prepare_pose(
                        camera_from_grasp_geometry=_strict_rigid_matrix(
                            camera_pose, field_name="DMTAC axial camera variant"),
                        base_from_grasp_geometry=_strict_rigid_matrix(
                            base_from_camera @ camera_pose, field_name="DMTAC axial base variant"),
                        raw_model_score=source.model_score,
                        raw_index=source.raw_index,
                        pose_source=GRASPGENX_DMTAC_AXIAL_POSE_SOURCE,
                        orientation_variant_axis=None, orientation_variant_angle_deg=0.,
                        support_clearance_world_z_m=0.)
                    if child is None:
                        continue
                    # Preserve any orientation/centering penalty of the parent,
                    # then apply only the existing new click-distance factor.
                    child = replace(child, axial_parent=source, axial_shift_m=shift,
                        score=source.score * math.exp(
                            (source.click_distance_m - child.click_distance_m) / 0.060))
                    child = admit_visible_scene(child)
                    if child is not None and child.capture_support_score > source.capture_support_score + 1e-9:
                        admitted_children.append(child)
                best = min(admitted_children,
                    key=lambda item: (-item.capture_support_score, item.axial_shift_m), default=None)
                axial_readmissions.append((source, best))
                if best is not None and not _pose_is_duplicate(
                    best.camera_from_grasp_geometry,
                    [item.camera_from_grasp_geometry for item in deduplicated]):
                    deduplicated.append(best)
            deduplicated.sort(key=lambda item: (-item.score, item.click_distance_m))

        # A family is a display-priority group, not an IK-equivalence class.
        # Nearby scored depths can differ in support clearance and arm feasibility.
        family_indices: list[int] = []
        for index, item in enumerate(deduplicated):
            if not any(
                self._same_candidate_family(
                    item.camera_from_grasp_geometry,
                    deduplicated[previous].camera_from_grasp_geometry,
                )
                for previous in family_indices
            ):
                family_indices.append(index)
        representative_indices, default_selected_indices = (
            timed_diversity(
                tuple(
                    deduplicated[index].camera_from_grasp_geometry
                    for index in family_indices
                ),
                maximum=maximum,
            )
        )
        default_selected_indices = [
            family_indices[index] for index in default_selected_indices
        ]
        geometry_from_tool = np.linalg.inv(
            self.solid_geometry.solid_pose(np.eye(4))
        )

        pool_indices = {id(item): index for index, item in enumerate(deduplicated)}
        for original, observed in reversed(surface_observation_replacements):
            if id(observed) in pool_indices:
                pool_indices[id(original)] = pool_indices[id(observed)]
        if refine_grasp_centers:
            counts["contact_orientation_pool_count"] = len(contact_orientation_readmissions)
            counts["contact_orientation_sources_json"] = json.dumps([
                {"parent_index": pool_indices[id(parent)], "pool_index": pool_indices[id(child)]}
                for parent, child in contact_orientation_readmissions], separators=(",", ":"))
            counts["height_variant_pool_count"] = len(height_readmissions)
            counts["height_variant_sources_json"] = json.dumps([
                {"parent_index": pool_indices.get(id(parent)), "raw_index": child.raw_index,
                 "pool_index": pool_indices[id(child)], "shift_m": child.height_shift_m,
                 "parent_support": parent.capture_height_support_score,
                 "support": child.capture_height_support_score}
                for parent, child in height_readmissions], separators=(",", ":"))
        if refine_grasp_axial_depth:
            counts["axial_variant_pool_count"] = sum(
                item.axial_parent is not None for item in deduplicated)
            counts["axial_variant_sources_json"] = json.dumps([
                {"parent_index": pool_indices[id(parent)], "parent_source": parent.pose_source,
                 "parent_support": parent.capture_support_score,
                 "pool_index": pool_indices.get(id(child)),
                 "shift_m": None if child is None else child.axial_shift_m,
                 "support": None if child is None else child.capture_support_score}
                for parent, child in axial_readmissions], separators=(",", ":"))
        if refine_grasp_centers:
            counts["centering_variant_pool_count"] = sum(
                item.pose_source == GRASPGENX_DMTAC_CENTERING_POSE_SOURCE for item in deduplicated)
            counts["centering_variant_sources_json"] = json.dumps([
                {"raw_index": index, "shift_m": shift, "pool_index": pool_indices.get(id(item))}
                for index, shift, item in centering_readmissions], separators=(",", ":"))

        def public_candidate(item: _RankedGraspCandidate) -> GraspGenXSelectionCandidate:
            return GraspGenXSelectionCandidate(
                score=float(item.score),
                model_score=float(item.model_score),
                click_distance_m=float(item.click_distance_m),
                gripper_width_m=float(item.gripper_width_m),
                camera_from_tcp=tuple(
                    float(value)
                    for value in (
                        item.camera_from_grasp_geometry @ geometry_from_tool
                    ).reshape(-1)
                ),
                base_from_tcp=tuple(
                    float(value)
                    for value in (
                        item.base_from_grasp_geometry @ geometry_from_tool
                    ).reshape(-1)
                ),
                visible_collision_free=True,
                pose_source=item.pose_source,
                orientation_variant_axis=item.orientation_variant_axis,
                orientation_variant_angle_deg=item.orientation_variant_angle_deg,
                support_clearance_world_z_m=item.support_clearance_world_z_m,
                symmetry_parent_index=pool_indices.get(id(item.symmetry_parent)),
                capture_support_score=item.capture_support_score,
                opposed_capture_support_score=item.opposed_capture_support_score,
                surface_alignment_score=item.surface_alignment_score,
                surface_alignment_observed_sides=item.surface_alignment_observed_sides,
                surface_opposition_score=item.surface_opposition_score,
                contact_height_support_score=item.contact_height_support_score,
                contact_height_center_shift_m=item.contact_height_center_shift_m,
                contact_height_evidence_weight=item.contact_height_evidence_weight,
                height_shift_basis=item.height_shift_basis,
                axial_parent_index=pool_indices.get(id(item.axial_parent)),
                axial_shift_m=item.axial_shift_m,
                capture_height_support_score=item.capture_height_support_score,
                capture_height_center_shift_m=item.capture_height_center_shift_m,
                height_parent_index=pool_indices.get(id(item.height_parent)),
                height_shift_m=item.height_shift_m,
                raw_index=item.raw_index,
            )

        reserved_variant_indices: list[int] = []
        # Without an embodiment selector, preserve the generic perception
        # shortlist and reserve bounded capacity for physically re-admitted
        # DMTAC variants.
        variant_return_limit = min(3, max(0, maximum - 1))
        support_variant_indices = [
            index
            for index, item in enumerate(deduplicated)
            if item.pose_source == "dmtac_support_clearance_variant_v1"
        ]
        reserved_variant_indices = support_variant_indices[
            : min(2, variant_return_limit)
        ]
        orientation_variant_slots = min(
            GRASPGENX_DMTAC_ORIENTATION_VARIANT_RETURN_LIMIT,
            variant_return_limit - len(reserved_variant_indices),
        )
        support_reserved_count = len(reserved_variant_indices)
        reserved_variant_indices.extend(
            index
            for index, item in enumerate(deduplicated)
            if item.pose_source == "dmtac_local_orientation_variant_v1"
            and index not in reserved_variant_indices
        )
        reserved_variant_indices = reserved_variant_indices[
            : support_reserved_count + orientation_variant_slots
        ]
        reserved_variant_set = set(reserved_variant_indices)
        combined_indices = list(reserved_variant_indices)
        combined_set = set(combined_indices)
        for index in default_selected_indices:
            if len(combined_indices) == maximum:
                break
            if (
                deduplicated[index].pose_source != "model_scored"
                and index not in reserved_variant_set
            ):
                continue
            if index not in combined_set:
                combined_indices.append(index)
                combined_set.add(index)
        selected_indices = combined_indices
        selected_set = set(selected_indices)
        for index in range(len(deduplicated)):
            if len(selected_indices) == maximum:
                break
            if deduplicated[index].pose_source != "model_scored":
                continue
            if index not in selected_set:
                selected_indices.append(index)
                selected_set.add(index)
        for index in range(len(deduplicated)):
            if len(selected_indices) == maximum:
                break
            if index not in selected_set:
                selected_indices.append(index)
                selected_set.add(index)
        selected_indices = tuple(sorted(selected_indices))
        counts["candidate_family_count"] = len(family_indices)
        counts["candidate_variant_count"] = len(deduplicated)
        counts["distinct_approach_count"] = len(representative_indices)
        # Semantic summaries must stay deterministic across equivalent requests.
        # Host wall times belong to the existing per-plan timing channel only.
        counts["candidate_geometry_call_counts_json"] = json.dumps(geometry_calls, separators=(",", ":"))
        if filter_summary is not None:
            filter_summary.update(counts)
        return _GraspGenXCandidatePool(
            candidates=tuple(public_candidate(item) for item in deduplicated),
            generic_indices=tuple(selected_indices),
            reserved_variant_indices=tuple(reserved_variant_indices),
            summary=MappingProxyType(dict(filter_summary) if filter_summary is not None else counts),
        )

    @staticmethod
    def _publish_pool(
        pool: _GraspGenXCandidatePool, *, target: tuple[float, float, float],
        maximum: int, selector: CandidateSelector | None,
    ) -> tuple[tuple[PointGraspCandidate, ...], dict[str, int | float | str | None]]:
        counts = dict(pool.summary)
        reserved = pool.reserved_variant_indices if selector is None else ()
        selected_indices = pool.generic_indices
        if selector is not None:
            selection = selector(pool.candidates, target, maximum)
            if not isinstance(selection, GraspGenXCandidateSelection):
                raise TypeError("candidate_selector must return GraspGenXCandidateSelection.")
            preferred_indices = tuple(selection.selected_indices)
            if (
                len(preferred_indices) > maximum
                or any(isinstance(index, bool) or not isinstance(index, int)
                    or not 0 <= index < len(pool.candidates) for index in preferred_indices)
                or len(set(preferred_indices)) != len(preferred_indices)
            ):
                raise ValueError("candidate_selector returned invalid indices.")
            for key, value in selection.diagnostics.items():
                if (not isinstance(key, str) or not key.startswith("arm_shortlist_")
                    or (value is not None and not isinstance(value, (int, float, str)))):
                    raise ValueError("candidate_selector diagnostics must be scalar arm_shortlist fields.")
                counts[key] = value
            # Public score order is independent of the advisory planning order.
            selected_indices = tuple(sorted(preferred_indices))
            published_ordinal = {index: ordinal for ordinal, index in enumerate(selected_indices)}
            counts[GRASP_PLANNING_ORDER_FIELD] = json.dumps(
                [published_ordinal[index] for index in preferred_indices], separators=(",", ":"))
            counts["arm_shortlist_planning_order_scope"] = GRASP_PLANNING_ORDER_SCOPE
        selected = tuple(pool.candidates[index] for index in selected_indices)
        counts["selected"] = len(selected)
        counts["min_selected_capture_support_score"] = min(
            (item.capture_support_score for item in selected), default=None)
        counts["min_selected_capture_height_support_score"] = min(
            (item.capture_height_support_score for item in selected), default=None)
        counts["selected_capture_geometry_json"] = json.dumps([
            {"raw_index": item.raw_index, "pose_source": item.pose_source,
             "orientation_variant_axis": item.orientation_variant_axis,
             "orientation_variant_angle_deg": item.orientation_variant_angle_deg,
             "proposal_width_m": item.gripper_width_m,
             "collision_opening_m": counts.get("collision_opening_m"),
             "capture_height_support_score": item.capture_height_support_score,
             "capture_height_center_shift_m": item.capture_height_center_shift_m,
             "capture_support_score": item.capture_support_score,
             "opposed_capture_support_score": item.opposed_capture_support_score,
             "surface_alignment_score": item.surface_alignment_score,
             "surface_alignment_observed_sides": item.surface_alignment_observed_sides,
             "surface_opposition_score": item.surface_opposition_score,
             "contact_height_support_score": item.contact_height_support_score,
             "contact_height_center_shift_m": item.contact_height_center_shift_m,
             "contact_height_evidence_weight": item.contact_height_evidence_weight,
             "height_shift_basis": item.height_shift_basis,
             "height_shift_m": item.height_shift_m,
             "T_reference_from_tool_tcp": item.base_from_tcp}
            for item in selected], separators=(",", ":"))
        if "height_variant_attempted" in counts:
            counts["height_variant_selected"] = sum(
                item.pose_source == GRASPGENX_DMTAC_HEIGHT_POSE_SOURCE for item in selected)
        if "centering_variant_attempted" in counts:
            counts["centering_variant_selected"] = sum(
                item.pose_source == GRASPGENX_DMTAC_CENTERING_POSE_SOURCE for item in selected)
        if "axial_variant_attempted" in counts:
            counts["axial_variant_selected"] = sum(
                item.pose_source == GRASPGENX_DMTAC_AXIAL_POSE_SOURCE for item in selected)
        for name, source in (
            ("orientation_variant", "dmtac_local_orientation_variant_v1"),
            ("finger_exchange", PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1),
            ("support_clearance_variant", "dmtac_support_clearance_variant_v1"),
            ("arm_side_variant", "dmtac_arm_side_approach_variant_v1"),
        ):
            counts[f"{name}_selected"] = sum(item.pose_source == source for item in selected)
        for name, source in (
            ("orientation_variant", "dmtac_local_orientation_variant_v1"),
            ("support_clearance_variant", "dmtac_support_clearance_variant_v1"),
        ):
            counts[f"{name}_reserved"] = sum(pool.candidates[index].pose_source == source for index in reserved)
        return tuple(PointGraspCandidate(**{
            field.name: getattr(item, field.name) for field in fields(PointGraspCandidate)
        }) for item in selected), counts

    @staticmethod
    def _same_candidate_family(first: np.ndarray, second: np.ndarray) -> bool:
        if (
            np.linalg.norm(first[:3, 3] - second[:3, 3])
            >= GRASPGENX_CANDIDATE_FAMILY_TRANSLATION_M
        ):
            return False
        cosine = float(
            np.clip(np.dot(first[:3, 2], second[:3, 2]), -1.0, 1.0)
        )
        approach_angle = math.acos(cosine)
        if approach_angle >= math.radians(
            GRASPGENX_CANDIDATE_FAMILY_APPROACH_DEG
        ):
            return False
        # Finger exchange is not a whole-robot symmetry. Keep opposite scored
        # closing axes for independent controller-TCP IK and path validation.
        closing_cosine = float(
            np.clip(np.dot(first[:3, 0], second[:3, 0]), -1.0, 1.0)
        )
        closing_axis_angle = math.acos(closing_cosine)
        return closing_axis_angle < math.radians(
            GRASPGENX_CANDIDATE_FAMILY_APPROACH_DEG
        )

    @staticmethod
    def _same_approach_cone(first: np.ndarray, second: np.ndarray) -> bool:
        cosine = float(
            np.clip(np.dot(first[:3, 2], second[:3, 2]), -1.0, 1.0)
        )
        return math.acos(cosine) < math.radians(
            GRASPGENX_CANDIDATE_APPROACH_DIVERSITY_DEG
        )

    @classmethod
    def _distinct_approach_first_indices(
        cls,
        poses: tuple[np.ndarray, ...],
        *,
        maximum: int,
    ) -> tuple[list[int], list[int]]:
        representative_indices: list[int] = []
        for index, pose in enumerate(poses):
            if any(
                cls._same_approach_cone(pose, poses[previous_index])
                for previous_index in representative_indices
            ):
                continue
            representative_indices.append(index)

        # If several approach cones survived the physical/proxy filters, keep
        # every one before filling remaining slots with scored variants.  A
        # half-budget here made five returned candidates look diverse in score
        # while all sharing one approach direction.
        selected_indices = representative_indices[:maximum]
        if len(selected_indices) < maximum:
            selected_set = set(selected_indices)
            represented = tuple(poses[index] for index in selected_indices)
            for index, pose in enumerate(poses):
                if index in selected_set or not any(
                    cls._same_approach_cone(pose, previous) for previous in represented
                ):
                    continue
                selected_indices.append(index)
                selected_set.add(index)
                if len(selected_indices) == maximum:
                    break
            for index in range(len(poses)):
                if len(selected_indices) == maximum:
                    break
                if index in selected_set:
                    continue
                selected_indices.append(index)
                selected_set.add(index)
                if len(selected_indices) == maximum:
                    break
        return representative_indices, selected_indices

    @property
    def solid_geometry(self) -> ParallelJawSolidGeometry:
        return self._solid_geometry


__all__ = [
    "DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1",
    "DMTAC_M_GRASPGENX_TERMINAL_TARGET_CONTACT_DEPTH_M",
    "GRASPGENX_ASSET_SCHEMA_V1",
    "GRASPGENX_CHECKPOINT_GIT_URL",
    "GRASPGENX_CHECKPOINT_REVIEWED_REVISION",
    "GRASPGENX_GIT_URL",
    "GRASPGENX_REVIEWED_REVISION",
    "GraspGenXCandidateSelection",
    "GraspGenXPointGraspModelSpec",
    "Sam2GraspGenXPointGraspPlanner",
]

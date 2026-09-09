"""SAM2 + GraspNet point-conditioned DMTAC M grasp planning."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import subprocess
import sys
import threading
import time
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from dataset_contracts.canonical_encoding import canonical_content_hash

from .dmtac_grasp_overlay import (
    DMTAC_M_FINGER_LENGTH_M,
    DMTAC_M_FINGER_THICKNESS_M,
    DMTAC_M_PAD_HEIGHT_M,
    DMTAC_M_PALM_FROM_TOOL_TCP_Z_M,
    DMTAC_M_TOOL_TCP_FROM_FLANGE_Z_M,
)
from .sam2_vgn_point_grasp import (
    DEFAULT_DMTAC_M_URDF,
    DEFAULT_DMTAC_M_URDF_SHA256,
    DMTAC_M_MAX_VISIBLE_COLLISION_POINTS,
    PARALLEL_JAW_TCP_UPWARD_SELECTION_V1,
    PointGraspCandidate,
    PointGraspResult,
    select_upward_parallel_jaw_tcp_pose,
)

GRASPNET_GIT_URL = "https://github.com/graspnet/graspnet-baseline.git"
GRASPNET_REVIEWED_REVISION = "280c215129f759ed8649cb4e89fc5dfee55f4f80"
GRASPNET_ASSET_SCHEMA_V1 = "dexteleop.perception_graspnet_point_grasp_assets.v1"
GRASPNET_LICENSE_ACKNOWLEDGEMENT_V1 = (
    "graspnet_noncommercial_research_only_local_download_no_team_redistribution.v1"
)
GRASPNET_PROJECT_AUTHORIZATION_ACKNOWLEDGEMENT_V2 = (
    "graspnet_project_authorized_use_and_redistribution.v2"
)
GRASPNET_ACCEPTED_AUTHORIZATION_ACKNOWLEDGEMENTS = frozenset(
    {
        GRASPNET_LICENSE_ACKNOWLEDGEMENT_V1,
        GRASPNET_PROJECT_AUTHORIZATION_ACKNOWLEDGEMENT_V2,
    }
)
GRASPNET_MODEL_FINGER_LENGTH_M = 0.060


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
        matrix[3], [0, 0, 0, 1], atol=1e-8
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


def _normalize_graspnet_rigid_matrix(
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
    if not np.allclose(
        projected.T @ projected,
        np.eye(3),
        atol=1e-9,
        rtol=0.0,
    ) or not np.isclose(np.linalg.det(projected), 1.0, atol=1e-9, rtol=0.0):
        raise ValueError(f"{field_name} rotation could not be normalized.")
    normalized = matrix.copy()
    normalized[:3, :3] = projected
    return _strict_rigid_matrix(normalized, field_name=field_name)


@dataclass(frozen=True)
class GraspNetPointGraspModelSpec:
    model_profile_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    source_revision: str
    model_type: str
    license_acknowledgement: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_profile_id",
            _required(self.model_profile_id, field_name="model_profile_id"),
        )
        checkpoint = Path(self.checkpoint_path).resolve(strict=True)
        if not checkpoint.is_file():
            raise ValueError("checkpoint_path must be an existing file.")
        object.__setattr__(self, "checkpoint_path", checkpoint)
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _lower_hex(
                self.checkpoint_sha256, field_name="checkpoint_sha256", length=64
            ),
        )
        object.__setattr__(
            self,
            "source_revision",
            _lower_hex(self.source_revision, field_name="source_revision", length=40),
        )
        if self.source_revision != GRASPNET_REVIEWED_REVISION:
            raise ValueError("GraspNet source revision is not the reviewed commit.")
        if self.model_type != "realsense":
            raise ValueError(
                "Only the configured GraspNet RealSense checkpoint type is supported."
            )
        if self.checkpoint_path.name != "checkpoint-rs.tar":
            raise ValueError("The GraspNet checkpoint must be named checkpoint-rs.tar.")
        if (
            self.license_acknowledgement
            not in GRASPNET_ACCEPTED_AUTHORIZATION_ACKNOWLEDGEMENTS
        ):
            raise ValueError(
                "GraspNet project authorization acknowledgement is required."
            )

    @classmethod
    def from_asset_bundle(cls, bundle_root: str | Path) -> GraspNetPointGraspModelSpec:
        root = Path(bundle_root).expanduser().resolve(strict=True)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "model_profile_id",
            "source",
            "checkpoint",
            "license_acknowledgement",
        }:
            raise ValueError("GraspNet asset manifest fields are invalid.")
        if manifest["schema_version"] != GRASPNET_ASSET_SCHEMA_V1:
            raise ValueError("Unsupported GraspNet asset manifest schema.")
        source = manifest["source"]
        checkpoint = manifest["checkpoint"]
        if not isinstance(source, dict) or set(source) != {"git_url", "revision"}:
            raise ValueError("GraspNet source identity is invalid.")
        if source["git_url"] != GRASPNET_GIT_URL:
            raise ValueError("GraspNet source URL is not reviewed.")
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "relative_path",
            "sha256",
            "model_type",
        }:
            raise ValueError("GraspNet checkpoint identity is invalid.")
        relative = Path(
            _required(
                checkpoint["relative_path"], field_name="checkpoint.relative_path"
            )
        )
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("GraspNet checkpoint must stay inside its local bundle.")
        path = (root / relative).resolve(strict=True)
        if not path.is_file() or root not in path.parents:
            raise ValueError("GraspNet checkpoint is not a regular bundle file.")
        expected = _lower_hex(
            checkpoint["sha256"], field_name="checkpoint.sha256", length=64
        )
        if _sha256_file(path) != expected:
            raise ValueError("GraspNet checkpoint hash mismatch.")
        return cls(
            model_profile_id=manifest["model_profile_id"],
            checkpoint_path=path,
            checkpoint_sha256=expected,
            source_revision=source["revision"],
            model_type=checkpoint["model_type"],
            license_acknowledgement=manifest["license_acknowledgement"],
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "model_profile_id": self.model_profile_id,
                "source": {
                    "git_url": GRASPNET_GIT_URL,
                    "revision": self.source_revision,
                },
                "checkpoint_sha256": self.checkpoint_sha256,
                "model_type": self.model_type,
                "license_acknowledgement": self.license_acknowledgement,
            },
            domain="dexteleop.perception.graspnet_point_grasp_model_spec.v1",
        )


class Sam2GraspNetPointGraspPlanner:
    """Lifecycle-owned GraspNet planner with hash-locked project assets."""

    backend_id = "graspnet"

    def __init__(
        self,
        *,
        model_spec: GraspNetPointGraspModelSpec,
        point_segmenter: object,
        device: str = "cuda",
        num_points: int = 20_000,
        num_views: int = 300,
        max_gripper_width_m: float = 0.070,
        max_target_surface_distance_m: float = 0.040,
        max_click_distance_m: float = 0.100,
        point_conditioned_workspace_margin_m: float = 0.120,
        max_target_width_centering_shift_m: float = 0.020,
        max_target_height_centering_shift_m: float = 0.030,
    ) -> None:
        if not isinstance(model_spec, GraspNetPointGraspModelSpec):
            raise TypeError("model_spec must be typed.")
        if not callable(getattr(point_segmenter, "segment_point", None)):
            raise TypeError("point_segmenter must expose segment_point().")
        self.model_spec = model_spec
        self.point_segmenter = point_segmenter
        self.device = _required(device, field_name="device")
        self.num_points = int(num_points)
        self.num_views = int(num_views)
        self.max_gripper_width_m = float(max_gripper_width_m)
        self.max_target_surface_distance_m = float(max_target_surface_distance_m)
        self.max_click_distance_m = float(max_click_distance_m)
        self.point_conditioned_workspace_margin_m = float(
            point_conditioned_workspace_margin_m
        )
        self.max_target_width_centering_shift_m = float(
            max_target_width_centering_shift_m
        )
        self.max_target_height_centering_shift_m = float(
            max_target_height_centering_shift_m
        )
        if not self.device.startswith("cuda"):
            raise ValueError(
                "Reviewed GraspNet pointnet2 inference requires a CUDA device."
            )
        if self.num_points != 20_000 or self.num_views != 300:
            raise ValueError(
                "Reviewed GraspNet profile requires 20000 points and 300 views."
            )
        if not 0.0 < self.max_gripper_width_m <= 0.070:
            raise ValueError("DMTAC M gripper width must be in (0, 0.070].")
        if not math.isclose(
            self.point_conditioned_workspace_margin_m,
            0.120,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Reviewed point-conditioned GraspNet workspace margin must be 0.120 m."
            )
        if not math.isclose(
            self.max_target_width_centering_shift_m,
            0.020,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Reviewed target width-centering shift must be 0.020 m.")
        if not math.isclose(
            self.max_target_height_centering_shift_m,
            0.030,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Reviewed target height-centering shift must be 0.030 m.")
        self._lock = threading.RLock()
        self._network: Any = None
        self._pred_decode: Any = None
        self._torch: Any = None
        self._started = False
        self._closed = False
        self._last_plan_timing_ms: dict[str, float] = {}
        self._last_target_mask: np.ndarray | None = None

    @property
    def last_plan_timing_ms(self) -> dict[str, float]:
        """Return passive wall-clock diagnostics for the most recent plan call."""

        with self._lock:
            return dict(self._last_plan_timing_ms)

    @property
    def last_target_mask(self) -> np.ndarray | None:
        """Return a diagnostic copy of the most recent SAM2 target mask."""

        with self._lock:
            return (
                None
                if self._last_target_mask is None
                else self._last_target_mask.copy()
            )

    def _record_plan_timing(self, timing: Mapping[str, float]) -> None:
        with self._lock:
            self._last_plan_timing_ms = {
                str(name): float(value) for name, value in timing.items()
            }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "backend": self.backend_id,
                "model_spec_hash": self.model_spec.semantic_content_hash,
                "sam2_segmenter_hash": _required(
                    getattr(self.point_segmenter, "semantic_content_hash", None),
                    field_name="point_segmenter.semantic_content_hash",
                ),
                "device": self.device,
                "num_points": self.num_points,
                "num_views": self.num_views,
                "sampling_seed": 0,
                "point_conditioned_workspace": {
                    "source": "sam2_mask_valid_depth_aabb",
                    "margin_m": self.point_conditioned_workspace_margin_m,
                    "minimum_points_before_full_scene_fallback": 128,
                    "collision_check_cloud": "full_aligned_depth",
                },
                "target_surface_distance_m": self.max_target_surface_distance_m,
                "target_width_centering": {
                    "source": "sam2_target_points_5_95_percentile_midpoint",
                    "maximum_shift_m": self.max_target_width_centering_shift_m,
                    "model_axis": "+y_opening",
                },
                "target_height_centering": {
                    "source": "sam2_target_points_5_95_percentile_midpoint",
                    "maximum_shift_m": self.max_target_height_centering_shift_m,
                    "model_axis": "+z_height",
                },
                "click_distance_m": self.max_click_distance_m,
                "gripper": {
                    "profile": "DMTAC_M",
                    "maximum_opening_m": self.max_gripper_width_m,
                    "maximum_visible_collision_points": DMTAC_M_MAX_VISIBLE_COLLISION_POINTS,
                    "tool_tcp_from_flange_m": [
                        0.0,
                        0.0,
                        DMTAC_M_TOOL_TCP_FROM_FLANGE_Z_M,
                    ],
                    "palm_from_tool_tcp_z_m": DMTAC_M_PALM_FROM_TOOL_TCP_Z_M,
                    "model_approach_axis": "+x",
                    "model_width_axis": "+y",
                    "tool_tcp_approach_axis": "+z",
                    "tool_tcp_width_axis": "+x",
                    "tool_tcp_up_axis": "+y",
                    "parallel_jaw_wrist_symmetry_deg": [0, 180],
                    "tcp_orientation_selection": (PARALLEL_JAW_TCP_UPWARD_SELECTION_V1),
                    "model_grasp_depth_mapping": {
                        "prediction_column": 3,
                        "model_finger_length_m": GRASPNET_MODEL_FINGER_LENGTH_M,
                        "model_output_origin": "grasp_seed",
                        "physical_palm_origin": "model_palm_finger_boundary",
                        "output_origin": "calibrated_actual_tool_tcp",
                    },
                },
                "asset_authorization": self.model_spec.license_acknowledgement,
                "simulation_urdf": DEFAULT_DMTAC_M_URDF,
                "simulation_urdf_sha256": DEFAULT_DMTAC_M_URDF_SHA256,
            },
            domain="dexteleop.perception.sam2_graspnet_point_grasp_planner.v1",
        )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Closed point-grasp planner cannot restart.")
            if self._started:
                return
            try:
                import pointnet2._ext  # noqa: F401 - verifies the required CUDA extension.
                import torch

                training_only_knn = types.ModuleType("knn_modules")

                def _training_only(*_args: Any, **_kwargs: Any) -> Any:
                    raise RuntimeError(
                        "Training-only GraspNet KNN was invoked during inference."
                    )

                training_only_knn.knn = _training_only
                previous_knn = sys.modules.get("knn_modules")
                sys.modules["knn_modules"] = training_only_knn
                try:
                    graspnet = importlib.import_module("graspnet")
                finally:
                    if previous_knn is None:
                        sys.modules.pop("knn_modules", None)
                    else:
                        sys.modules["knn_modules"] = previous_knn
                GraspNet = graspnet.GraspNet
                pred_decode = graspnet.pred_decode
            except ImportError as exc:
                raise RuntimeError(
                    "GraspNet is not installed; run install_point_grasp_models.sh "
                    "with --with-graspnet."
                ) from exc
            revision = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(graspnet.__file__).resolve().parent),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip()
            if revision != self.model_spec.source_revision:
                raise RuntimeError(
                    "Installed GraspNet revision differs from the local manifest."
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GraspNet CUDA deployment requested but CUDA is unavailable."
                )
            selected_device = torch.device(self.device)
            network = GraspNet(
                input_feature_dim=0,
                num_view=self.num_views,
                num_angle=12,
                num_depth=4,
                cylinder_radius=0.05,
                hmin=-0.02,
                hmax_list=[0.01, 0.02, 0.03, 0.04],
                is_training=False,
            ).to(selected_device)
            checkpoint = torch.load(
                self.model_spec.checkpoint_path,
                map_location=selected_device,
                weights_only=False,
            )
            if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
                raise RuntimeError(
                    "GraspNet checkpoint does not contain model_state_dict."
                )
            network.load_state_dict(checkpoint["model_state_dict"], strict=True)
            network.eval()
            self._network = network
            self._pred_decode = pred_decode
            self._torch = torch
            self._started = True

    def close(self) -> None:
        with self._lock:
            self._network = None
            self._pred_decode = None
            self._torch = None
            self._started = False
            self._closed = True

    def plan(
        self,
        image: Image.Image,
        depth_m: np.ndarray,
        *,
        pixel_uv: tuple[int, int],
        intrinsics: Mapping[str, float | int],
        base_from_camera: np.ndarray,
        max_candidates: int,
    ) -> PointGraspResult:
        plan_started = time.perf_counter()
        timing_ms: dict[str, float] = {}
        self._record_plan_timing(timing_ms)
        if not self._started or self._closed:
            raise RuntimeError("Point-grasp planner is not running.")
        rgb = image.convert("RGB")
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.shape != (rgb.height, rgb.width) or not np.all(np.isfinite(depth)):
            raise ValueError("Depth must be a finite aligned HxW meter image.")
        if isinstance(max_candidates, bool) or not 1 <= int(max_candidates) <= 5:
            raise ValueError("max_candidates must be in [1, 5].")
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
            values["fx"] <= 0
            or values["fy"] <= 0
            or not all(math.isfinite(value) for value in values.values())
        ):
            raise ValueError("Camera intrinsics are invalid.")
        transform_base_camera = _rigid_matrix(
            base_from_camera, field_name="base_from_camera"
        )
        sam2_started = time.perf_counter()
        try:
            mask = np.asarray(
                self.point_segmenter.segment_point(rgb, pixel_uv=(u, v)),
                dtype=bool,
            )
        finally:
            timing_ms["sam2_inference"] = (time.perf_counter() - sam2_started) * 1000.0
            timing_ms["plan_total"] = (time.perf_counter() - plan_started) * 1000.0
            self._record_plan_timing(timing_ms)
        if mask.shape != depth.shape or not mask[v, u]:
            raise RuntimeError("SAM2 returned an invalid point mask.")
        with self._lock:
            self._last_target_mask = mask.copy()
        valid = (depth > 0.0) & (depth <= 10.0)
        rows, columns = np.nonzero(valid)
        if len(rows) < 128:
            raise RuntimeError(
                "Aligned depth has too few valid scene points for GraspNet."
            )
        z = depth[rows, columns].astype(np.float32)
        scene_points = np.column_stack(
            (
                (columns - values["cx"]) * z / values["fx"],
                (rows - values["cy"]) * z / values["fy"],
                z,
            )
        ).astype(np.float32)
        target_valid = valid & mask
        target_rows, target_columns = np.nonzero(target_valid)
        if len(target_rows) < 16:
            raise RuntimeError("Clicked SAM2 region has too few valid depth points.")
        target_z = depth[target_rows, target_columns].astype(np.float32)
        target_points = np.column_stack(
            (
                (target_columns - values["cx"]) * target_z / values["fx"],
                (target_rows - values["cy"]) * target_z / values["fy"],
                target_z,
            )
        ).astype(np.float32)
        target_depth = self._target_depth(depth, mask, u=u, v=v)
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
        inference_points = self._point_conditioned_workspace(
            scene_points,
            target_points,
        )
        graspnet_started = time.perf_counter()
        try:
            predictions = self._infer(inference_points)
        finally:
            timing_ms["graspnet_inference"] = (
                time.perf_counter() - graspnet_started
            ) * 1000.0
            timing_ms["plan_total"] = (time.perf_counter() - plan_started) * 1000.0
            self._record_plan_timing(timing_ms)
        ranking_started = time.perf_counter()
        try:
            candidates = self._rank(
                predictions,
                target_points=target_points,
                target_camera=point_camera[:3],
                base_from_camera=transform_base_camera,
                depth=depth,
                intrinsics=values,
                maximum=int(max_candidates),
            )
        finally:
            timing_ms["candidate_ranking_and_visible_collision"] = (
                time.perf_counter() - ranking_started
            ) * 1000.0
            timing_ms["plan_total"] = (time.perf_counter() - plan_started) * 1000.0
            self._record_plan_timing(timing_ms)
        if not candidates:
            raise RuntimeError(
                "No visible collision-free DMTAC M GraspNet candidate belongs to the clicked region."
            )
        mask_rows, mask_columns = np.nonzero(mask)
        return PointGraspResult(
            target_point_camera_m=tuple(float(value) for value in point_camera[:3]),
            target_point_base_m=tuple(float(value) for value in point_base[:3]),
            target_mask_bbox_xyxy=(
                int(mask_columns.min()),
                int(mask_rows.min()),
                int(mask_columns.max()) + 1,
                int(mask_rows.max()) + 1,
            ),
            candidates=candidates,
        )

    def _infer(self, scene_points: np.ndarray) -> np.ndarray:
        torch = self._torch
        network = self._network
        pred_decode = self._pred_decode
        if torch is None or network is None or pred_decode is None:
            raise RuntimeError("GraspNet network is unavailable.")
        generator = np.random.default_rng(0)
        replace = len(scene_points) < self.num_points
        indices = generator.choice(
            len(scene_points), size=self.num_points, replace=replace
        )
        sampled = np.ascontiguousarray(scene_points[indices][None], dtype=np.float32)
        end_points = {
            "point_clouds": torch.from_numpy(sampled).to(torch.device(self.device))
        }
        with torch.no_grad():
            decoded = pred_decode(network(end_points))[0]
        predictions = decoded.detach().cpu().numpy().astype(np.float64, copy=False)
        if predictions.ndim != 2 or predictions.shape[1] != 17:
            raise RuntimeError("GraspNet returned an unexpected prediction shape.")
        return predictions

    def _point_conditioned_workspace(
        self,
        scene_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Crop inference around the clicked SAM2 region in metric 3D.

        GraspNet's official path applies a workspace mask before fixed-size
        sampling. Uniformly sampling an entire room/simulator view and using
        the clicked object only for post-hoc ranking removes small targets from
        the inference cloud. The complete depth image remains authoritative for
        the later visible-collision gate.
        """

        scene = np.asarray(scene_points, dtype=np.float32)
        target = np.asarray(target_points, dtype=np.float32)
        if (
            scene.ndim != 2
            or target.ndim != 2
            or scene.shape[1:] != (3,)
            or target.shape[1:] != (3,)
            or len(target) < 16
        ):
            raise ValueError(
                "Point-conditioned GraspNet workspace expects Nx3 scene and target points."
            )
        margin = self.point_conditioned_workspace_margin_m
        lower = np.min(target, axis=0) - margin
        upper = np.max(target, axis=0) + margin
        within = np.all((scene >= lower) & (scene <= upper), axis=1)
        cropped = np.ascontiguousarray(scene[within], dtype=np.float32)
        if len(cropped) < 128:
            return np.ascontiguousarray(scene, dtype=np.float32)
        return cropped

    def _rank(
        self,
        predictions: np.ndarray,
        *,
        target_points: np.ndarray,
        target_camera: np.ndarray,
        base_from_camera: np.ndarray,
        depth: np.ndarray,
        intrinsics: Mapping[str, float],
        maximum: int,
    ) -> tuple[PointGraspCandidate, ...]:
        if len(target_points) > 50_000:
            indices = np.linspace(0, len(target_points) - 1, 50_000, dtype=np.int64)
            target_points = target_points[indices]
        tree = cKDTree(target_points)
        centers = predictions[:, 13:16]
        surface_distances = tree.query(centers, k=1, workers=1)[0]

        # GraspNet: +X approach, +Y opening, +Z height. DMTAC tool_tcp:
        # +Z approach, +X opening, +Y height.
        grasp_from_tcp_rotation = np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        ranked: list[tuple[float, float, float, float, np.ndarray, np.ndarray]] = []
        for prediction, surface_distance in zip(
            predictions, surface_distances, strict=True
        ):
            if not np.all(np.isfinite(prediction)):
                continue
            model_score = min(1.0, max(0.0, float(prediction[0])))
            width = float(prediction[1])
            grasp_depth = float(prediction[3])
            if not 0.005 <= width <= self.max_gripper_width_m:
                continue
            if not 0.010 <= grasp_depth <= 0.040:
                continue
            if float(surface_distance) > self.max_target_surface_distance_m:
                continue
            camera_from_grasp = np.eye(4, dtype=np.float64)
            camera_from_grasp[:3, :3] = prediction[4:13].reshape(3, 3)
            camera_from_grasp[:3, 3] = prediction[13:16]
            try:
                camera_from_grasp = _rigid_matrix(
                    camera_from_grasp, field_name="GraspNet camera_from_grasp"
                )
            except ValueError:
                continue
            # GraspNet seeds lie on observed surfaces and can be biased toward
            # one jaw or one end of an object.  Center both axes orthogonal to
            # approach using robust clicked-SAM2 extents before mapping to the
            # physical DMTAC TCP.  This uses no simulator object pose or
            # grasp-outcome oracle.
            for axis_index, maximum_shift in (
                (1, self.max_target_width_centering_shift_m),
                (2, self.max_target_height_centering_shift_m),
            ):
                axis = camera_from_grasp[:3, axis_index]
                coordinates = (target_points - camera_from_grasp[:3, 3]) @ axis
                low, high = np.percentile(coordinates, (5.0, 95.0))
                center_shift = float(
                    np.clip(
                        (low + high) * 0.5,
                        -maximum_shift,
                        maximum_shift,
                    )
                )
                camera_from_grasp[:3, 3] += axis * center_shift
            try:
                camera_from_grasp = _normalize_graspnet_rigid_matrix(
                    camera_from_grasp,
                    field_name="GraspNet camera_from_grasp",
                )
            except ValueError:
                continue
            # GraspNet's translation is the grasp seed.  In its official
            # collision geometry the fingers occupy local approach X in
            # [depth - 0.06, depth] and the palm ends at depth - 0.06.
            # First resolve the GraspNet seed to its model palm/finger boundary,
            # then continue to the calibrated DMTAC actual tool_tcp.  Omitting
            # the second translation makes a palm pose masquerade as tool_tcp.
            grasp_from_tcp = np.eye(4, dtype=np.float64)
            grasp_from_tcp[:3, :3] = grasp_from_tcp_rotation
            grasp_from_tcp[:3, 3] = (
                grasp_depth
                - GRASPNET_MODEL_FINGER_LENGTH_M
                - DMTAC_M_PALM_FROM_TOOL_TCP_Z_M,
                0.0,
                0.0,
            )
            camera_from_tcp = camera_from_grasp @ grasp_from_tcp
            base_from_tcp = base_from_camera @ camera_from_tcp
            camera_from_tcp, base_from_tcp = select_upward_parallel_jaw_tcp_pose(
                camera_from_tcp,
                base_from_tcp,
            )
            camera_from_tcp = _strict_rigid_matrix(
                camera_from_tcp,
                field_name="GraspNet camera_from_tcp candidate",
            )
            base_from_tcp = _strict_rigid_matrix(
                base_from_tcp,
                field_name="GraspNet base_from_tcp candidate",
            )
            click_distance = float(
                np.linalg.norm(camera_from_grasp[:3, 3] - target_camera)
            )
            if click_distance > self.max_click_distance_m:
                continue
            if not self._visible_collision_free(
                camera_from_tcp, width, depth, intrinsics
            ):
                continue
            score = (
                model_score
                * math.exp(-float(surface_distance) / 0.025)
                * math.exp(-click_distance / 0.060)
            )
            ranked.append(
                (
                    score,
                    model_score,
                    click_distance,
                    width,
                    camera_from_tcp,
                    base_from_tcp,
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[2]))
        selected: list[tuple[float, float, float, float, np.ndarray, np.ndarray]] = []
        for item in ranked:
            pose = item[4]
            duplicate = any(
                np.linalg.norm(pose[:3, 3] - previous[4][:3, 3]) < 0.015
                and self._rotation_distance_radians(pose[:3, :3], previous[4][:3, :3])
                < math.radians(15.0)
                for previous in selected
            )
            if duplicate:
                continue
            selected.append(item)
            if len(selected) == maximum:
                break
        return tuple(
            PointGraspCandidate(
                score=float(score),
                model_score=float(model_score),
                click_distance_m=float(click_distance),
                gripper_width_m=float(width),
                camera_from_tcp=tuple(
                    float(value) for value in camera_from_tcp.reshape(-1)
                ),
                base_from_tcp=tuple(
                    float(value) for value in base_from_tcp.reshape(-1)
                ),
                visible_collision_free=True,
            )
            for score, model_score, click_distance, width, camera_from_tcp, base_from_tcp in selected
        )

    @staticmethod
    def _rotation_distance_radians(first: np.ndarray, second: np.ndarray) -> float:
        relative = first.T @ second
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        return math.acos(cosine)

    @staticmethod
    def _target_depth(depth: np.ndarray, mask: np.ndarray, *, u: int, v: int) -> float:
        if depth[v, u] > 0:
            return float(depth[v, u])
        radius = 5
        ys = slice(max(0, v - radius), min(depth.shape[0], v + radius + 1))
        xs = slice(max(0, u - radius), min(depth.shape[1], u + radius + 1))
        values = depth[ys, xs]
        valid = values[(values > 0) & mask[ys, xs]]
        if not len(valid):
            raise RuntimeError(
                "Clicked region has no valid aligned depth within 5 pixels."
            )
        return float(np.median(valid))

    @staticmethod
    def _visible_collision_free(
        camera_from_tcp: np.ndarray,
        width: float,
        depth: np.ndarray,
        intrinsics: Mapping[str, float],
    ) -> bool:
        step = max(1, int(math.sqrt(depth.size / 100_000)))
        rows, columns = np.nonzero(depth[::step, ::step] > 0)
        rows = rows * step
        columns = columns * step
        z = depth[rows, columns].astype(np.float64)
        points = np.column_stack(
            (
                (columns - intrinsics["cx"]) * z / intrinsics["fx"],
                (rows - intrinsics["cy"]) * z / intrinsics["fy"],
                z,
            )
        )
        tcp_from_camera = np.linalg.inv(camera_from_tcp)
        local = points @ tcp_from_camera[:3, :3].T + tcp_from_camera[:3, 3]
        height = DMTAC_M_PAD_HEIGHT_M
        finger = DMTAC_M_FINGER_THICKNESS_M
        palm_z = DMTAC_M_PALM_FROM_TOOL_TCP_Z_M
        fingertip_z = palm_z + DMTAC_M_FINGER_LENGTH_M
        within_height = np.abs(local[:, 1]) < height / 2.0
        forward = (local[:, 2] > palm_z) & (local[:, 2] < fingertip_z)
        left = (local[:, 0] > width / 2.0) & (local[:, 0] < width / 2.0 + finger)
        right = (local[:, 0] < -width / 2.0) & (local[:, 0] > -width / 2.0 - finger)
        palm = (
            (local[:, 2] > palm_z - finger)
            & (local[:, 2] <= palm_z)
            & (np.abs(local[:, 0]) < width / 2.0 + finger)
        )
        return (
            int(np.count_nonzero(within_height & (forward & (left | right) | palm)))
            <= DMTAC_M_MAX_VISIBLE_COLLISION_POINTS
        )


__all__ = [
    "GRASPNET_ASSET_SCHEMA_V1",
    "GRASPNET_GIT_URL",
    "GRASPNET_LICENSE_ACKNOWLEDGEMENT_V1",
    "GRASPNET_MODEL_FINGER_LENGTH_M",
    "GRASPNET_PROJECT_AUTHORIZATION_ACKNOWLEDGEMENT_V2",
    "GRASPNET_REVIEWED_REVISION",
    "GraspNetPointGraspModelSpec",
    "Sam2GraspNetPointGraspPlanner",
]

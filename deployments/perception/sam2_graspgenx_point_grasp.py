"""SAM2 + GraspGenX point-conditioned DMTAC M grasp proposals."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_contracts.canonical_encoding import canonical_content_hash

from .dmtac_grasp_overlay import (
    DMTAC_M_FINGER_LENGTH_M,
    DMTAC_M_PAD_HEIGHT_M,
    DMTAC_M_PALM_FROM_TOOL_TCP_Z_M,
    DMTAC_M_TOOL_TCP_FROM_FLANGE_Z_M,
)
from .graspgenx_client import GraspGenXInferenceBackend
from .sam2_vgn_point_grasp import (
    DEFAULT_DMTAC_M_URDF,
    DEFAULT_DMTAC_M_URDF_SHA256,
    DMTAC_M_MAX_VISIBLE_COLLISION_POINTS,
    PARALLEL_JAW_TCP_UPWARD_SELECTION_V1,
    PointGraspCandidate,
    PointGraspResult,
    Sam2VgnPointGraspPlanner,
    select_upward_parallel_jaw_tcp_pose,
)

GRASPGENX_GIT_URL = "https://github.com/NVlabs/GraspGenX.git"
GRASPGENX_REVIEWED_REVISION = "b9429097728cb1c430dd78b92edf17ba318aad03"
GRASPGENX_CHECKPOINT_GIT_URL = "https://huggingface.co/adithyamurali/GraspGenXModel"
GRASPGENX_CHECKPOINT_REVIEWED_REVISION = "7c834043c11a11417e31d6d5ea9355801e40a2c1"
GRASPGENX_ASSET_SCHEMA_V1 = "dexteleop.perception_graspgenx_point_grasp_assets.v1"
GRASPGENX_SOURCE_LICENSE = "Apache-2.0"
GRASPGENX_CHECKPOINT_LICENSE = "NVIDIA-Open-Model-License"

# DMTAC M's inner closing volume in the calibrated tool convention.  GraspGenX
# expects +Z approach and +X closing at the palm/finger boundary.  The linkage
# keeps the opposing pads parallel, so this is a parallel_2f embodiment.
DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1: dict[str, object] = {
    "extents_open": (0.070, DMTAC_M_PAD_HEIGHT_M, DMTAC_M_FINGER_LENGTH_M),
    "offset_open": (0.0, 0.0, DMTAC_M_FINGER_LENGTH_M / 2.0),
    "extents_mid": (0.035, DMTAC_M_PAD_HEIGHT_M, DMTAC_M_FINGER_LENGTH_M),
    "offset_mid": (0.0, 0.0, DMTAC_M_FINGER_LENGTH_M / 2.0),
    "gripper_type": 0,
    "fingertip_depth": DMTAC_M_FINGER_LENGTH_M,
}
DMTAC_M_GRASPGENX_WIDTH_CLEARANCE_M = 0.006


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
        server_top_k: int = 100,
        score_threshold: float = -1.0,
        max_gripper_width_m: float = 0.070,
        max_click_distance_m: float = 0.100,
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
        if isinstance(server_top_k, bool) or not 1 <= int(server_top_k) <= 1_000:
            raise ValueError("server_top_k must be in [1, 1000].")
        if not 0.0 < float(max_gripper_width_m) <= 0.070:
            raise ValueError("DMTAC M gripper width must be in (0, 0.070].")
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
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._last_target_mask: np.ndarray | None = None
        self._last_plan_timing_ms: dict[str, float] = {}

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
                "gripper": {
                    "profile": "DMTAC_M",
                    "sweep_volume_v1": DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
                    "model_base_from_tool_tcp_z_m": -DMTAC_M_PALM_FROM_TOOL_TCP_Z_M,
                    "maximum_opening_m": self.max_gripper_width_m,
                    "width_clearance_m": DMTAC_M_GRASPGENX_WIDTH_CLEARANCE_M,
                    "maximum_visible_collision_points": DMTAC_M_MAX_VISIBLE_COLLISION_POINTS,
                    "tool_tcp_from_flange_m": [
                        0.0,
                        0.0,
                        DMTAC_M_TOOL_TCP_FROM_FLANGE_Z_M,
                    ],
                    "parallel_jaw_wrist_symmetry_deg": [0, 180],
                    "tcp_orientation_selection": PARALLEL_JAW_TCP_UPWARD_SELECTION_V1,
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
        self,
        image: Image.Image,
        depth_m: np.ndarray,
        *,
        pixel_uv: tuple[int, int],
        intrinsics: Mapping[str, float | int],
        base_from_camera: np.ndarray,
        max_candidates: int,
    ) -> PointGraspResult:
        started = time.perf_counter()
        timing: dict[str, float] = {}
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
            values["fx"] <= 0.0
            or values["fy"] <= 0.0
            or not all(math.isfinite(value) for value in values.values())
        ):
            raise ValueError("Camera intrinsics are invalid.")
        transform_base_camera = _rigid_matrix(
            base_from_camera, field_name="base_from_camera"
        )
        stage_started = time.perf_counter()
        mask = np.asarray(
            self.point_segmenter.segment_point(rgb, pixel_uv=(u, v)), dtype=bool
        )
        timing["sam2_inference"] = (time.perf_counter() - stage_started) * 1000.0
        if mask.shape != depth.shape or not mask[v, u]:
            raise RuntimeError("SAM2 returned an invalid point mask.")
        with self._lock:
            self._last_target_mask = mask.copy()
        target_points = Sam2VgnPointGraspPlanner._masked_points_camera(
            depth, mask, values
        ).astype(np.float32)
        if len(target_points) > 50_000:
            indices = np.linspace(0, len(target_points) - 1, 50_000, dtype=np.int64)
            target_points = np.ascontiguousarray(target_points[indices])
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
        stage_started = time.perf_counter()
        grasps, scores = self.client.infer_object(
            target_points,
            sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
            planner=self.planner,
            num_grasps=self.num_grasps,
            score_threshold=self.score_threshold,
            top_k=self.server_top_k,
        )
        timing["graspgenx_inference"] = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        candidates = self._rank(
            grasps,
            scores,
            target_points=target_points,
            target_camera=point_camera[:3],
            base_from_camera=transform_base_camera,
            depth=depth,
            intrinsics=values,
            maximum=int(max_candidates),
        )
        timing["candidate_ranking_and_visible_collision"] = (
            time.perf_counter() - stage_started
        ) * 1000.0
        timing["plan_total"] = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._last_plan_timing_ms = timing
        if not candidates:
            raise RuntimeError(
                "No visible collision-free DMTAC M GraspGenX candidate belongs "
                "to the clicked region."
            )
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
            candidates=candidates,
        )

    def _rank(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        *,
        target_points: np.ndarray,
        target_camera: np.ndarray,
        base_from_camera: np.ndarray,
        depth: np.ndarray,
        intrinsics: Mapping[str, float],
        maximum: int,
    ) -> tuple[PointGraspCandidate, ...]:
        model_base_from_tcp = np.eye(4, dtype=np.float64)
        model_base_from_tcp[2, 3] = -DMTAC_M_PALM_FROM_TOOL_TCP_Z_M
        ranked: list[tuple[float, float, float, float, np.ndarray, np.ndarray]] = []
        for camera_from_model_base, raw_score in zip(grasps, scores, strict=True):
            camera_from_tcp = camera_from_model_base @ model_base_from_tcp
            base_from_tcp = base_from_camera @ camera_from_tcp
            camera_from_tcp, base_from_tcp = select_upward_parallel_jaw_tcp_pose(
                camera_from_tcp, base_from_tcp
            )
            tcp_from_camera = np.linalg.inv(camera_from_tcp)
            local = target_points @ tcp_from_camera[:3, :3].T + tcp_from_camera[:3, 3]
            low, high = np.percentile(local, (5.0, 95.0), axis=0)
            target_center = (low + high) * 0.5
            requested_width = (
                float(high[0] - low[0]) + DMTAC_M_GRASPGENX_WIDTH_CLEARANCE_M
            )
            if requested_width > self.max_gripper_width_m:
                continue
            width = max(0.005, requested_width)
            palm_z = DMTAC_M_PALM_FROM_TOOL_TCP_Z_M
            fingertip_z = palm_z + DMTAC_M_FINGER_LENGTH_M
            if (
                abs(float(target_center[0])) > width / 2.0
                or abs(float(target_center[1])) > DMTAC_M_PAD_HEIGHT_M / 2.0
                or not palm_z <= float(target_center[2]) <= fingertip_z
            ):
                continue
            click_distance = float(
                np.linalg.norm(camera_from_model_base[:3, 3] - target_camera)
            )
            if click_distance > self.max_click_distance_m:
                continue
            if not Sam2VgnPointGraspPlanner._visible_collision_free(
                camera_from_tcp, width, depth, intrinsics
            ):
                continue
            model_score = float(np.clip(raw_score, 0.0, 1.0))
            score = model_score * math.exp(-click_distance / 0.060)
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
            if any(
                np.linalg.norm(pose[:3, 3] - previous[4][:3, 3]) < 0.015
                and self._rotation_distance_radians(pose[:3, :3], previous[4][:3, :3])
                < math.radians(15.0)
                for previous in selected
            ):
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


__all__ = [
    "DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1",
    "GRASPGENX_ASSET_SCHEMA_V1",
    "GRASPGENX_CHECKPOINT_GIT_URL",
    "GRASPGENX_CHECKPOINT_REVIEWED_REVISION",
    "GRASPGENX_GIT_URL",
    "GRASPGENX_REVIEWED_REVISION",
    "GraspGenXPointGraspModelSpec",
    "Sam2GraspGenXPointGraspPlanner",
]

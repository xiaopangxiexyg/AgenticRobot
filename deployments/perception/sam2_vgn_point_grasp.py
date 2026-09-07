"""Point-conditioned RGB-D grasp planning with reviewed SAM2 and VGN sources."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash

from .dmtac_grasp_overlay import (
    DMTAC_M_FINGER_LENGTH_M,
    DMTAC_M_FINGER_THICKNESS_M,
    DMTAC_M_PAD_HEIGHT_M,
    DMTAC_M_PALM_FROM_TOOL_TCP_Z_M,
    DMTAC_M_TOOL_TCP_FROM_FLANGE_Z_M,
)

VGN_GIT_URL = "https://github.com/ethz-asl/vgn.git"
POINT_GRASP_ASSET_SCHEMA_V1 = "dexteleop.perception_point_grasp_assets.v1"
VGN_REVIEWED_REVISION = "d7af0622433f52ae88ebe81533f12b46b33e951a"
DEFAULT_DMTAC_M_URDF = (
    "assets/rm65_inspire_gripper_v3/"
    "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
)
DEFAULT_DMTAC_M_URDF_SHA256 = (
    "457a2b92c81cc4f6f2f1ada63f1ed7a4a659209c4af4aba7d2dcd8403e3b0620"
)
DMTAC_M_MAX_VISIBLE_COLLISION_POINTS = 400
DMTAC_M_TARGET_OPENING_CENTER_MAX_SHIFT_M = 0.035
DMTAC_M_TARGET_HEIGHT_CENTER_MAX_SHIFT_M = 0.050
PARALLEL_JAW_TCP_UPWARD_SELECTION_V1 = "maximize_reference_plus_z_dot_tool_tcp_plus_y"


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


def select_upward_parallel_jaw_tcp_pose(
    camera_from_tcp: np.ndarray,
    reference_from_tcp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the upward TCP from one equivalent parallel-jaw pose pair.

    Rotating DMTAC M by 180 degrees about tool-TCP +Z (its approach axis)
    merely exchanges the two fingers.  Tool-TCP +Y is its height/wrist-up
    axis, so select the member whose +Y axis has the larger projection on the
    reference frame's gravity-up +Z axis.  Exact horizontal ties preserve the
    model pose.  Both frame representations receive the same local rotation.
    """

    camera = _rigid_matrix(camera_from_tcp, field_name="camera_from_tcp")
    reference = _rigid_matrix(reference_from_tcp, field_name="reference_from_tcp")
    if float(reference[2, 1]) < 0.0:
        wrist_flip = np.diag((-1.0, -1.0, 1.0, 1.0))
        camera = camera @ wrist_flip
        reference = reference @ wrist_flip
    return camera, reference


def _rotation_distance_radians(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


@dataclass(frozen=True)
class VgnPointGraspModelSpec:
    model_profile_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    source_revision: str
    model_type: str

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
        if self.source_revision != VGN_REVIEWED_REVISION:
            raise ValueError("VGN source revision is not the reviewed commit.")
        if self.model_type != "conv":
            raise ValueError(
                "Only the configured VGN conv checkpoint type is supported."
            )
        if self.checkpoint_path.name != "vgn_conv.pth":
            raise ValueError("The VGN checkpoint must be named vgn_conv.pth.")

    @classmethod
    def from_asset_bundle(cls, bundle_root: str | Path) -> VgnPointGraspModelSpec:
        root = Path(bundle_root).expanduser().resolve(strict=True)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "model_profile_id",
            "source",
            "checkpoint",
        }:
            raise ValueError("Point-grasp asset manifest fields are invalid.")
        if manifest["schema_version"] != POINT_GRASP_ASSET_SCHEMA_V1:
            raise ValueError("Unsupported point-grasp asset manifest schema.")
        source = manifest["source"]
        checkpoint = manifest["checkpoint"]
        if not isinstance(source, dict) or set(source) != {"git_url", "revision"}:
            raise ValueError("Point-grasp source identity is invalid.")
        if source["git_url"] != VGN_GIT_URL:
            raise ValueError("Point-grasp source URL is not reviewed.")
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "relative_path",
            "sha256",
            "model_type",
        }:
            raise ValueError("Point-grasp checkpoint identity is invalid.")
        relative = Path(
            _required(
                checkpoint["relative_path"], field_name="checkpoint.relative_path"
            )
        )
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "Point-grasp checkpoint must stay inside its asset bundle."
            )
        path = (root / relative).resolve(strict=True)
        if not path.is_file() or root not in path.parents:
            raise ValueError("Point-grasp checkpoint is not a regular bundle file.")
        expected = _lower_hex(
            checkpoint["sha256"], field_name="checkpoint.sha256", length=64
        )
        if _sha256_file(path) != expected:
            raise ValueError("Point-grasp checkpoint hash mismatch.")
        return cls(
            model_profile_id=manifest["model_profile_id"],
            checkpoint_path=path,
            checkpoint_sha256=expected,
            source_revision=source["revision"],
            model_type=checkpoint["model_type"],
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "model_profile_id": self.model_profile_id,
                "source": {"git_url": VGN_GIT_URL, "revision": self.source_revision},
                "checkpoint_sha256": self.checkpoint_sha256,
                "model_type": self.model_type,
            },
            domain="dexteleop.perception.vgn_point_grasp_model_spec.v1",
        )


@dataclass(frozen=True)
class PointGraspCandidate:
    score: float
    model_score: float
    click_distance_m: float
    gripper_width_m: float
    camera_from_tcp: tuple[float, ...]
    base_from_tcp: tuple[float, ...]
    visible_collision_free: bool


@dataclass(frozen=True)
class PointGraspResult:
    target_point_camera_m: tuple[float, float, float]
    target_point_base_m: tuple[float, float, float]
    target_mask_bbox_xyxy: tuple[int, int, int, int]
    candidates: tuple[PointGraspCandidate, ...]


class Sam2VgnPointGraspPlanner:
    """Lifecycle-owned, read-only VGN planner using an existing SAM2 segmenter."""

    backend_id = "vgn"

    def __init__(
        self,
        *,
        model_spec: VgnPointGraspModelSpec,
        point_segmenter: object,
        device: str = "cuda",
        cube_size_m: float = 0.30,
        resolution: int = 40,
        quality_threshold: float = 0.90,
        max_gripper_width_m: float = 0.070,
        max_click_distance_m: float = 0.080,
    ) -> None:
        if not isinstance(model_spec, VgnPointGraspModelSpec):
            raise TypeError("model_spec must be typed.")
        if not callable(getattr(point_segmenter, "segment_point", None)):
            raise TypeError("point_segmenter must expose segment_point().")
        self.model_spec = model_spec
        self.point_segmenter = point_segmenter
        self.device = _required(device, field_name="device")
        self.cube_size_m = float(cube_size_m)
        self.resolution = int(resolution)
        self.quality_threshold = float(quality_threshold)
        self.max_gripper_width_m = float(max_gripper_width_m)
        self.max_click_distance_m = float(max_click_distance_m)
        if self.cube_size_m != 0.30 or self.resolution != 40:
            raise ValueError(
                "Reviewed VGN checkpoint requires a 0.30 m, 40-voxel cube."
            )
        if self.quality_threshold != 0.90:
            raise ValueError(
                "The reviewed VGN detector uses a fixed 0.90 quality threshold."
            )
        if not 0.0 < self.max_gripper_width_m <= 0.070:
            raise ValueError("DMTAC M gripper width must be in (0, 0.070].")
        self._lock = threading.RLock()
        self._network: Any = None
        self._torch: Any = None
        self._started = False
        self._closed = False

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "model_spec_hash": self.model_spec.semantic_content_hash,
                "sam2_segmenter_hash": _required(
                    getattr(self.point_segmenter, "semantic_content_hash", None),
                    field_name="point_segmenter.semantic_content_hash",
                ),
                "device": self.device,
                "cube_size_m": self.cube_size_m,
                "resolution": self.resolution,
                "quality_threshold": self.quality_threshold,
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
                    "finger_length_m": DMTAC_M_FINGER_LENGTH_M,
                    "target_opening_center_max_shift_m": (
                        DMTAC_M_TARGET_OPENING_CENTER_MAX_SHIFT_M
                    ),
                    "target_height_center_max_shift_m": (
                        DMTAC_M_TARGET_HEIGHT_CENTER_MAX_SHIFT_M
                    ),
                    "model_grasp_width_axis": "+y",
                    "tool_tcp_width_axis": "+x",
                    "tool_tcp_approach_axis": "+z",
                    "tool_tcp_up_axis": "+y",
                    "parallel_jaw_wrist_symmetry_deg": [0, 180],
                    "tcp_orientation_selection": (PARALLEL_JAW_TCP_UPWARD_SELECTION_V1),
                },
                "simulation_urdf": DEFAULT_DMTAC_M_URDF,
                "simulation_urdf_sha256": DEFAULT_DMTAC_M_URDF_SHA256,
            },
            domain="dexteleop.perception.sam2_vgn_point_grasp_planner.v1",
        )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Closed point-grasp planner cannot restart.")
            if self._started:
                return
            try:
                import torch
                import vgn
                from vgn.networks import load_network
            except ImportError as exc:
                raise RuntimeError(
                    "VGN is not installed; run scripts/install_point_grasp_models.sh."
                ) from exc
            revision = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(vgn.__file__).resolve().parent),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip()
            if revision != self.model_spec.source_revision:
                raise RuntimeError(
                    "Installed VGN revision differs from the asset manifest."
                )
            selected_device = torch.device(self.device)
            if selected_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "VGN CUDA deployment requested but CUDA is unavailable."
                )
            self._network = load_network(
                self.model_spec.checkpoint_path, selected_device
            )
            self._network.eval()
            self._torch = torch
            self._started = True

    def close(self) -> None:
        with self._lock:
            self._network = None
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
            or not all(math.isfinite(x) for x in values.values())
        ):
            raise ValueError("Camera intrinsics are invalid.")
        transform_base_camera = _rigid_matrix(
            base_from_camera, field_name="base_from_camera"
        )
        mask = np.asarray(
            self.point_segmenter.segment_point(rgb, pixel_uv=(u, v)), dtype=bool
        )
        if mask.shape != depth.shape or not mask[v, u]:
            raise RuntimeError("SAM2 returned an invalid point mask.")
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
        base_from_cube = np.eye(4, dtype=np.float64)
        base_from_cube[:3, 3] = point_base[:3] - self.cube_size_m / 2.0
        camera_from_cube = np.linalg.inv(transform_base_camera) @ base_from_cube
        tsdf = self._projective_tsdf(depth, mask, values, camera_from_cube)
        target_points_camera = self._masked_points_camera(
            depth,
            mask,
            values,
        )
        with self._lock:
            grasps, scores = self._detect(tsdf)
        candidates = self._rank(
            grasps,
            scores,
            base_from_cube=base_from_cube,
            camera_from_cube=camera_from_cube,
            target_camera=point_camera[:3],
            target_points_camera=target_points_camera,
            depth=depth,
            intrinsics=values,
            maximum=int(max_candidates),
        )
        if not candidates:
            raise RuntimeError(
                "No visible collision-free DMTAC M grasp was found near the clicked region."
            )
        rows, columns = np.nonzero(mask)
        return PointGraspResult(
            target_point_camera_m=tuple(float(x) for x in point_camera[:3]),
            target_point_base_m=tuple(float(x) for x in point_base[:3]),
            target_mask_bbox_xyxy=(
                int(columns.min()),
                int(rows.min()),
                int(columns.max()) + 1,
                int(rows.max()) + 1,
            ),
            candidates=candidates,
        )

    def _detect(self, tsdf: np.ndarray) -> tuple[list[Any], np.ndarray]:
        """Run the reviewed VGN conv network without importing its ROS visualizer."""

        torch = self._torch
        network = self._network
        if torch is None or network is None:
            raise RuntimeError("Point-grasp planner network is unavailable.")
        tensor = torch.from_numpy(tsdf).unsqueeze(0).to(torch.device(self.device))
        with torch.no_grad():
            quality, quaternion, width = network(tensor)
        quality = quality.cpu().squeeze().numpy()
        quaternion = quaternion.cpu().squeeze().numpy()
        width = width.cpu().squeeze().numpy()

        quality = ndimage.gaussian_filter(quality, sigma=1.0, mode="nearest")
        signed = tsdf.squeeze()
        outside = signed > 0.5
        inside = (signed > 1e-3) & (signed < 0.5)
        valid = ndimage.binary_dilation(outside, iterations=2, mask=~inside)
        quality[~valid] = 0.0
        quality[(width < 1.33) | (width > 9.33)] = 0.0
        quality[quality < self.quality_threshold] = 0.0
        maxima = ndimage.maximum_filter(quality, size=4)
        indices = np.argwhere((quality == maxima) & (quality > 0.0))

        voxel_size = self.cube_size_m / self.resolution
        detected: list[Any] = []
        scores: list[float] = []
        for i, j, k in indices:
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = Rotation.from_quat(quaternion[:, i, j, k]).as_matrix()
            matrix[:3, 3] = np.asarray([i, j, k], dtype=np.float64) * voxel_size
            detected.append(
                SimpleNamespace(
                    width=float(width[i, j, k]) * voxel_size,
                    pose=SimpleNamespace(as_matrix=lambda value=matrix: value.copy()),
                )
            )
            scores.append(float(quality[i, j, k]))
        order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
        return [detected[int(index)] for index in order], np.asarray(scores)[order]

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

    def _projective_tsdf(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: Mapping[str, float],
        camera_from_cube: np.ndarray,
    ) -> np.ndarray:
        indices = np.indices((self.resolution,) * 3, dtype=np.float32).reshape(3, -1).T
        cube_points = (indices + 0.5) * (self.cube_size_m / self.resolution)
        camera_points = (
            cube_points @ camera_from_cube[:3, :3].T + camera_from_cube[:3, 3]
        )
        z = camera_points[:, 2]
        projected_u = np.rint(
            intrinsics["fx"] * camera_points[:, 0] / np.maximum(z, 1e-6)
            + intrinsics["cx"]
        ).astype(np.int64)
        projected_v = np.rint(
            intrinsics["fy"] * camera_points[:, 1] / np.maximum(z, 1e-6)
            + intrinsics["cy"]
        ).astype(np.int64)
        valid = (
            (z > 0)
            & (projected_u >= 0)
            & (projected_u < depth.shape[1])
            & (projected_v >= 0)
            & (projected_v < depth.shape[0])
        )
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] &= mask[
            projected_v[valid_indices], projected_u[valid_indices]
        ]
        valid_indices = np.flatnonzero(valid)
        measured = np.zeros_like(z)
        measured[valid_indices] = depth[
            projected_v[valid_indices], projected_u[valid_indices]
        ]
        valid &= measured > 0
        truncation = 4.0 * self.cube_size_m / self.resolution
        grid = np.zeros(len(z), dtype=np.float32)
        sdf = measured[valid] - z[valid]
        grid[valid] = (np.clip(sdf / truncation, -1.0, 1.0) + 1.0) * 0.5
        return grid.reshape(1, self.resolution, self.resolution, self.resolution)

    @staticmethod
    def _masked_points_camera(
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: Mapping[str, float],
    ) -> np.ndarray:
        rows, columns = np.nonzero(mask & (depth > 0.0))
        if len(rows) < 16:
            raise RuntimeError(
                "Clicked SAM2 region has fewer than 16 valid metric-depth points."
            )
        if len(rows) > 50_000:
            selected = np.linspace(0, len(rows) - 1, 50_000, dtype=np.int64)
            rows = rows[selected]
            columns = columns[selected]
        z = depth[rows, columns].astype(np.float64)
        return np.column_stack(
            (
                (columns - intrinsics["cx"]) * z / intrinsics["fx"],
                (rows - intrinsics["cy"]) * z / intrinsics["fy"],
                z,
            )
        )

    @staticmethod
    def _center_tcp_on_target(
        camera_from_tcp: np.ndarray,
        target_points_camera: np.ndarray,
    ) -> np.ndarray:
        centered = camera_from_tcp.copy()
        for axis_index, maximum_shift in (
            (0, DMTAC_M_TARGET_OPENING_CENTER_MAX_SHIFT_M),
            (1, DMTAC_M_TARGET_HEIGHT_CENTER_MAX_SHIFT_M),
        ):
            axis = centered[:3, axis_index]
            coordinates = (target_points_camera - centered[:3, 3]) @ axis
            low, high = np.percentile(coordinates, (5.0, 95.0))
            shift = float(np.clip((low + high) * 0.5, -maximum_shift, maximum_shift))
            centered[:3, 3] += axis * shift
        return centered

    def _rank(
        self,
        grasps: Any,
        scores: Any,
        *,
        base_from_cube: np.ndarray,
        camera_from_cube: np.ndarray,
        target_camera: np.ndarray,
        target_points_camera: np.ndarray,
        depth: np.ndarray,
        intrinsics: Mapping[str, float],
        maximum: int,
    ) -> tuple[PointGraspCandidate, ...]:
        # VGN uses +Y as jaw width and +Z as approach, with its origin at the
        # palm/finger boundary. DMTAC M uses +X as jaw width and +Z as
        # approach, but its calibrated actual tool_tcp is 42.999 mm forward of
        # that boundary. Both the axis mapping and physical translation are
        # therefore required.
        grasp_from_tcp = np.eye(4, dtype=np.float64)
        grasp_from_tcp[:3, :3] = np.asarray(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64
        )
        grasp_from_tcp[:3, 3] = np.asarray(
            [0.0, 0.0, -DMTAC_M_PALM_FROM_TOOL_TCP_Z_M],
            dtype=np.float64,
        )
        base_from_camera = base_from_cube @ np.linalg.inv(camera_from_cube)
        ranked = []
        for grasp, raw_score in zip(grasps, scores, strict=True):
            width = float(grasp.width)
            if not 0.005 <= width <= self.max_gripper_width_m:
                continue
            cube_from_grasp = np.asarray(grasp.pose.as_matrix(), dtype=np.float64)
            camera_from_tcp = camera_from_cube @ cube_from_grasp @ grasp_from_tcp
            camera_from_tcp = self._center_tcp_on_target(
                camera_from_tcp,
                target_points_camera,
            )
            base_from_tcp = base_from_camera @ camera_from_tcp
            camera_from_tcp, base_from_tcp = select_upward_parallel_jaw_tcp_pose(
                camera_from_tcp,
                base_from_tcp,
            )
            distance = float(np.linalg.norm(camera_from_tcp[:3, 3] - target_camera))
            if distance > self.max_click_distance_m:
                continue
            collision_free = self._visible_collision_free(
                camera_from_tcp, width, depth, intrinsics
            )
            if not collision_free:
                continue
            model_score = min(1.0, max(0.0, float(raw_score)))
            score = model_score * math.exp(-distance / 0.05)
            ranked.append(
                (score, model_score, distance, width, camera_from_tcp, base_from_tcp)
            )
        ranked.sort(key=lambda item: (-item[0], item[2]))
        selected = []
        for item in ranked:
            pose = item[4]
            duplicate = any(
                np.linalg.norm(pose[:3, 3] - previous[4][:3, 3]) < 0.015
                and _rotation_distance_radians(pose[:3, :3], previous[4][:3, :3])
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
                click_distance_m=float(distance),
                gripper_width_m=float(width),
                camera_from_tcp=tuple(float(x) for x in camera_from_tcp.reshape(-1)),
                base_from_tcp=tuple(float(x) for x in base_from_tcp.reshape(-1)),
                visible_collision_free=True,
            )
            for score, model_score, distance, width, camera_from_tcp, base_from_tcp in selected
        )

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
        # Actual DMTAC M visible envelope expressed about the calibrated
        # tool_tcp, not the palm/finger boundary used by VGN.
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
    "DEFAULT_DMTAC_M_URDF",
    "DEFAULT_DMTAC_M_URDF_SHA256",
    "DMTAC_M_MAX_VISIBLE_COLLISION_POINTS",
    "PARALLEL_JAW_TCP_UPWARD_SELECTION_V1",
    "POINT_GRASP_ASSET_SCHEMA_V1",
    "VGN_GIT_URL",
    "VGN_REVIEWED_REVISION",
    "PointGraspCandidate",
    "PointGraspResult",
    "Sam2VgnPointGraspPlanner",
    "VgnPointGraspModelSpec",
    "select_upward_parallel_jaw_tcp_pose",
]

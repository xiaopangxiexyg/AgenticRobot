from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SOURCE_CAMERA_MODEL_CHOICES = ("mujoco", "undistort-alpha0", "undistort-alpha1")
DEFAULT_DISTORTION_SOURCE_CAMERA_MODEL = "undistort-alpha0"


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RGB distortion postprocess requires opencv-python.") from exc
    return cv2


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _scaled_raw_camera_matrix(
    intrinsics: Mapping[str, object],
    *,
    width: int,
    height: int,
) -> np.ndarray | None:
    intr_w = _float_or_none(intrinsics.get("width"))
    intr_h = _float_or_none(intrinsics.get("height"))
    fx = _float_or_none(intrinsics.get("fx"))
    fy = _float_or_none(intrinsics.get("fy"))
    ppx = _float_or_none(intrinsics.get("ppx"))
    ppy = _float_or_none(intrinsics.get("ppy"))
    if intr_w is None or intr_h is None or fx is None or fy is None or ppx is None or ppy is None:
        return None
    if intr_w <= 0.0 or intr_h <= 0.0 or fx <= 0.0 or fy <= 0.0:
        return None
    scale_x = float(width) / float(intr_w)
    scale_y = float(height) / float(intr_h)
    return np.asarray(
        [
            [fx * scale_x, 0.0, ppx * scale_x],
            [0.0, fy * scale_y, ppy * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _dist_coeffs(intrinsics: Mapping[str, object]) -> np.ndarray:
    coeffs = intrinsics.get("coeffs")
    if not isinstance(coeffs, (list, tuple)):
        return np.zeros((5, 1), dtype=np.float64)
    values: list[float] = []
    for value in coeffs:
        number = _float_or_none(value)
        values.append(0.0 if number is None else float(number))
    if not values:
        values = [0.0] * 5
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def _has_nonzero_distortion(dist_coeffs: np.ndarray, *, eps: float = 1e-12) -> bool:
    return bool(np.any(np.abs(np.asarray(dist_coeffs, dtype=np.float64)) > float(eps)))


def _mujoco_source_camera_matrix(
    camera_metadata: Mapping[str, object],
    *,
    width: int,
    height: int,
) -> np.ndarray | None:
    fovy_deg = _float_or_none(camera_metadata.get("fovy_deg"))
    if fovy_deg is None or fovy_deg <= 0.0:
        return None
    fovy_rad = math.radians(float(fovy_deg))
    focal = float(height) / (2.0 * math.tan(fovy_rad / 2.0))
    if not math.isfinite(focal) or focal <= 0.0:
        return None
    return np.asarray(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _source_camera_matrix(
    *,
    source_camera_model: str,
    raw_camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    camera_metadata: Mapping[str, object],
    width: int,
    height: int,
) -> np.ndarray | None:
    cv2 = _require_cv2()
    if source_camera_model == "mujoco":
        return _mujoco_source_camera_matrix(camera_metadata, width=width, height=height)
    if source_camera_model == "undistort-alpha0":
        return cv2.getOptimalNewCameraMatrix(
            raw_camera_matrix,
            dist_coeffs,
            (int(width), int(height)),
            0.0,
            (int(width), int(height)),
        )[0]
    if source_camera_model == "undistort-alpha1":
        return cv2.getOptimalNewCameraMatrix(
            raw_camera_matrix,
            dist_coeffs,
            (int(width), int(height)),
            1.0,
            (int(width), int(height)),
        )[0]
    raise ValueError(
        f"source_camera_model must be one of {SOURCE_CAMERA_MODEL_CHOICES}, got {source_camera_model!r}."
    )


@dataclass(frozen=True)
class RgbDistortionMap:
    camera_name: str
    map_x: np.ndarray
    map_y: np.ndarray
    raw_camera_matrix: np.ndarray
    source_camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    outside_fraction: float

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.map_x.shape[0]), int(self.map_x.shape[1])


class RealRawRgbDistorter:
    """Warp pinhole sim RGB into the source dataset's raw RealSense distortion model."""

    def __init__(
        self,
        *,
        camera_intrinsics_by_name: Mapping[str, Mapping[str, object]],
        camera_metadata_by_name: Mapping[str, Mapping[str, object]],
        source_camera_model: str = "mujoco",
    ) -> None:
        if source_camera_model not in SOURCE_CAMERA_MODEL_CHOICES:
            raise ValueError(
                f"source_camera_model must be one of {SOURCE_CAMERA_MODEL_CHOICES}, got {source_camera_model!r}."
            )
        self.camera_intrinsics_by_name = {
            str(name): dict(value)
            for name, value in camera_intrinsics_by_name.items()
            if isinstance(value, Mapping)
        }
        self.camera_metadata_by_name = {
            str(name): dict(value)
            for name, value in camera_metadata_by_name.items()
            if isinstance(value, Mapping)
        }
        self.source_camera_model = str(source_camera_model)
        self._maps: dict[tuple[str, int, int], RgbDistortionMap | None] = {}

    def _build_map(self, camera_name: str, *, width: int, height: int) -> RgbDistortionMap | None:
        intrinsics = self.camera_intrinsics_by_name.get(str(camera_name))
        metadata = self.camera_metadata_by_name.get(str(camera_name), {})
        if not isinstance(intrinsics, Mapping):
            return None
        raw_k = _scaled_raw_camera_matrix(intrinsics, width=width, height=height)
        if raw_k is None:
            return None
        dist = _dist_coeffs(intrinsics)
        if not _has_nonzero_distortion(dist):
            return None
        source_k = _source_camera_matrix(
            source_camera_model=self.source_camera_model,
            raw_camera_matrix=raw_k,
            dist_coeffs=dist,
            camera_metadata=metadata,
            width=width,
            height=height,
        )
        if source_k is None:
            return None

        cv2 = _require_cv2()
        xs, ys = np.meshgrid(
            np.arange(int(width), dtype=np.float32),
            np.arange(int(height), dtype=np.float32),
        )
        raw_pixels = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
        source_pixels = cv2.undistortPoints(raw_pixels, raw_k, dist, P=source_k).reshape(height, width, 2)
        map_x = np.ascontiguousarray(source_pixels[..., 0].astype(np.float32))
        map_y = np.ascontiguousarray(source_pixels[..., 1].astype(np.float32))
        outside = (map_x < 0.0) | (map_x > float(width - 1)) | (map_y < 0.0) | (map_y > float(height - 1))
        return RgbDistortionMap(
            camera_name=str(camera_name),
            map_x=map_x,
            map_y=map_y,
            raw_camera_matrix=raw_k,
            source_camera_matrix=np.asarray(source_k, dtype=np.float64),
            dist_coeffs=dist,
            outside_fraction=float(np.count_nonzero(outside)) / float(outside.size),
        )

    def map_for(self, camera_name: str, *, width: int, height: int) -> RgbDistortionMap | None:
        key = (str(camera_name), int(width), int(height))
        if key not in self._maps:
            self._maps[key] = self._build_map(str(camera_name), width=int(width), height=int(height))
        return self._maps[key]

    def distort(self, camera_name: str, image: np.ndarray) -> np.ndarray:
        frame = np.asarray(image)
        if frame.ndim < 2:
            raise ValueError(f"RGB distortion expects an image array, got shape {frame.shape!r}.")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        distortion_map = self.map_for(str(camera_name), width=width, height=height)
        if distortion_map is None:
            return image
        cv2 = _require_cv2()
        return cv2.remap(
            frame,
            distortion_map.map_x,
            distortion_map.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def distort_images(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {str(name): self.distort(str(name), image) for name, image in images.items()}

    def metadata(self) -> dict[str, object]:
        cameras: dict[str, object] = {}
        for (camera_name, width, height), distortion_map in sorted(self._maps.items()):
            if distortion_map is None:
                cameras[camera_name] = {
                    "enabled": False,
                    "width": int(width),
                    "height": int(height),
                    "reason": "missing intrinsics or zero distortion",
                }
                continue
            cameras[camera_name] = {
                "enabled": True,
                "width": int(width),
                "height": int(height),
                "raw_camera_matrix": distortion_map.raw_camera_matrix.tolist(),
                "source_camera_matrix": distortion_map.source_camera_matrix.tolist(),
                "dist_coeffs": distortion_map.dist_coeffs.reshape(-1).tolist(),
                "outside_fraction": float(distortion_map.outside_fraction),
            }
        return {
            "enabled": True,
            "format": "dexteleop_sim_rgb_to_realsense_raw_distortion_v1",
            "model": "opencv_undistortPoints_inverse_remap",
            "source_camera_model": self.source_camera_model,
            "semantics": (
                "RGB frames were rendered by MuJoCo as pinhole images, then remapped so output "
                "pixels follow the source dataset RealSense raw color intrinsics and distortion coefficients."
            ),
            "cameras": cameras,
        }


def render_intrinsics_for_rgb_distortion(
    camera_intrinsics_by_name: Mapping[str, Mapping[str, object]],
    *,
    source_camera_model: str = DEFAULT_DISTORTION_SOURCE_CAMERA_MODEL,
) -> dict[str, dict[str, object]]:
    """Return source pinhole intrinsics used to render before raw distortion remap.

    MuJoCo only consumes the vertical focal length through `fovy`, but keeping the
    full matrix in metadata makes the effective approximation explicit.
    """

    out = {
        str(name): dict(value)
        for name, value in camera_intrinsics_by_name.items()
        if isinstance(value, Mapping)
    }
    if source_camera_model == "mujoco":
        return out
    if source_camera_model not in SOURCE_CAMERA_MODEL_CHOICES:
        raise ValueError(
            f"source_camera_model must be one of {SOURCE_CAMERA_MODEL_CHOICES}, got {source_camera_model!r}."
        )

    for camera_name, intrinsics in list(out.items()):
        width = int(intrinsics.get("width", 0) or 0)
        height = int(intrinsics.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            continue
        raw_k = _scaled_raw_camera_matrix(intrinsics, width=width, height=height)
        if raw_k is None:
            continue
        dist = _dist_coeffs(intrinsics)
        if not _has_nonzero_distortion(dist):
            continue
        source_k = _source_camera_matrix(
            source_camera_model=source_camera_model,
            raw_camera_matrix=raw_k,
            dist_coeffs=dist,
            camera_metadata={},
            width=width,
            height=height,
        )
        if source_k is None:
            continue
        updated = dict(intrinsics)
        updated.update(
            {
                "fx": float(source_k[0, 0]),
                "fy": float(source_k[1, 1]),
                "ppx": float(source_k[0, 2]),
                "ppy": float(source_k[1, 2]),
                "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                "source": f"sim_rgb_distortion_render:{source_camera_model}",
                "raw_real_intrinsics": dict(intrinsics),
            }
        )
        out[camera_name] = updated
    return out

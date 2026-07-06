from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def require_cv2():
    try:
        import cv2  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "real_undistort requires opencv-python in the active Python environment."
        ) from exc
    return cv2


def _float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(number)


def _int(value: Any, *, name: str) -> int:
    number = _float(value, name=name)
    rounded = int(round(number))
    if rounded <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return rounded


@dataclass(frozen=True)
class CameraIntrinsics:
    """OpenCV pinhole intrinsics plus distortion coefficients."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, camera_key: str = "") -> "CameraIntrinsics":
        prefix = f"{camera_key}." if camera_key else ""
        coeffs_raw = value.get("coeffs", ())
        if coeffs_raw is None:
            coeffs_raw = ()
        if not isinstance(coeffs_raw, Sequence) or isinstance(coeffs_raw, str | bytes):
            raise ValueError(f"{prefix}coeffs must be a numeric sequence.")
        coeffs = tuple(_float(item, name=f"{prefix}coeffs") for item in coeffs_raw)
        if not coeffs:
            coeffs = (0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            width=_int(value.get("width"), name=f"{prefix}width"),
            height=_int(value.get("height"), name=f"{prefix}height"),
            fx=_float(value.get("fx"), name=f"{prefix}fx"),
            fy=_float(value.get("fy"), name=f"{prefix}fy"),
            ppx=_float(value.get("ppx"), name=f"{prefix}ppx"),
            ppy=_float(value.get("ppy"), name=f"{prefix}ppy"),
            coeffs=coeffs,
        )

    def camera_matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [self.fx, 0.0, self.ppx],
                [0.0, self.fy, self.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def dist_coeffs(self) -> np.ndarray:
        return np.asarray(self.coeffs, dtype=np.float64).reshape(-1, 1)

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        width = int(width)
        height = int(height)
        if width == self.width and height == self.height:
            return self
        sx = float(width) / float(self.width)
        sy = float(height) / float(self.height)
        return CameraIntrinsics(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            ppx=self.ppx * sx,
            ppy=self.ppy * sy,
            coeffs=self.coeffs,
        )

    def with_zero_distortion(self, matrix: np.ndarray | None = None) -> "CameraIntrinsics":
        if matrix is None:
            matrix = self.camera_matrix()
        matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        return CameraIntrinsics(
            width=self.width,
            height=self.height,
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            ppx=float(matrix[0, 2]),
            ppy=float(matrix[1, 2]),
            coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "ppx": float(self.ppx),
            "ppy": float(self.ppy),
            "coeffs": [float(value) for value in self.coeffs],
        }


@dataclass(frozen=True)
class UndistortMaps:
    source: CameraIntrinsics
    output: CameraIntrinsics
    map1: np.ndarray
    map2: np.ndarray
    roi: tuple[int, int, int, int]
    alpha: float

    def remap(self, frame: np.ndarray, *, interpolation: int | None = None) -> np.ndarray:
        cv2 = require_cv2()
        if interpolation is None:
            interpolation = cv2.INTER_LINEAR
        image = np.asarray(frame)
        if image.shape[1] != self.source.width or image.shape[0] != self.source.height:
            raise ValueError(
                "frame size does not match undistort maps: "
                f"frame={image.shape[1]}x{image.shape[0]}, "
                f"maps={self.source.width}x{self.source.height}"
            )
        return cv2.remap(
            image,
            self.map1,
            self.map2,
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
        )


def build_undistort_maps(
    intrinsics: CameraIntrinsics,
    *,
    alpha: float = 0.0,
    map_type: str = "fixed",
) -> UndistortMaps:
    """Precompute OpenCV remap tables for fast per-frame undistortion."""

    cv2 = require_cv2()
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    size = (int(intrinsics.width), int(intrinsics.height))
    source_k = intrinsics.camera_matrix()
    dist = intrinsics.dist_coeffs()
    output_k, roi = cv2.getOptimalNewCameraMatrix(
        source_k,
        dist,
        size,
        alpha,
        size,
        centerPrincipalPoint=False,
    )
    cv_map_type = cv2.CV_16SC2 if str(map_type) == "fixed" else cv2.CV_32FC1
    map1, map2 = cv2.initUndistortRectifyMap(
        source_k,
        dist,
        None,
        output_k,
        size,
        cv_map_type,
    )
    return UndistortMaps(
        source=intrinsics,
        output=intrinsics.with_zero_distortion(output_k),
        map1=map1,
        map2=map2,
        roi=tuple(int(value) for value in roi),
        alpha=alpha,
    )


def load_realsense_intrinsics(devices: Mapping[str, Any]) -> dict[str, CameraIntrinsics]:
    out: dict[str, CameraIntrinsics] = {}
    for camera_key, device in devices.items():
        if not isinstance(device, Mapping):
            continue
        intrinsics = device.get("intrinsics")
        if isinstance(intrinsics, Mapping):
            out[str(camera_key)] = CameraIntrinsics.from_mapping(
                intrinsics,
                camera_key=str(camera_key),
            )
    return out


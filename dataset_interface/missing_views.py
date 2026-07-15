from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import MissingViewConfig
from .metadata import DexTeleopDatasetIndex


@dataclass(frozen=True)
class CameraResolution:
    height: int
    width: int
    sources: tuple[str, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def _feature_resolution(info: Mapping[str, Any], camera_key: str) -> tuple[int, int] | None:
    features = info.get("features")
    if not isinstance(features, Mapping):
        return None
    candidates = (
        f"observation.images.{camera_key}",
        f"observation.images.{camera_key}.augmented",
        str(camera_key),
    )
    for key in candidates:
        feature = features.get(key)
        if not isinstance(feature, Mapping):
            continue
        shape = feature.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            continue
        values = tuple(int(value) for value in shape)
        if values[-1] == 3:
            height, width = values[0], values[1]
        elif values[0] == 3:
            height, width = values[1], values[2]
        else:
            continue
        if height > 0 and width > 0:
            return (height, width)
    return None


def _video_resolution(path: Path) -> tuple[int, int]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "resolving reference-dataset camera dimensions requires opencv-python"
        ) from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open reference video for resolution: {path}")
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)))
    finally:
        capture.release()
    if height < 1 or width < 1:
        raise RuntimeError(f"reference video has invalid resolution {width}x{height}: {path}")
    return (height, width)


def _dataset_camera_resolutions(index: DexTeleopDatasetIndex) -> dict[str, tuple[int, int]]:
    resolutions: dict[str, tuple[int, int]] = {}
    for camera_key in index.declared_camera_keys:
        resolution = _feature_resolution(index.info, camera_key)
        if resolution is None:
            video_path = index.representative_video_path(camera_key)
            if video_path is None or not video_path.is_file():
                continue
            resolution = _video_resolution(video_path)
        resolutions[camera_key] = resolution
    return resolutions


class MissingViewResolver:
    """Resolve model-selected missing-view dimensions before DataLoader reads."""

    def __init__(self, dataset_index: DexTeleopDatasetIndex, config: MissingViewConfig):
        self.config = config
        declarations: dict[str, list[tuple[tuple[int, int], str]]] = {}

        for camera_key, resolution in config.explicit_resolutions.items():
            declarations.setdefault(camera_key, []).append(
                (tuple(resolution), f"explicit:{camera_key}")
            )

        indexes = [dataset_index]
        for root in config.reference_dataset_roots:
            resolved_root = Path(root).expanduser().resolve()
            if resolved_root == dataset_index.dataset_root:
                continue
            indexes.append(DexTeleopDatasetIndex(resolved_root))
        for index in indexes:
            for camera_key, resolution in _dataset_camera_resolutions(index).items():
                declarations.setdefault(camera_key, []).append(
                    (resolution, f"dataset:{index.dataset_root}")
                )

        self._resolutions: dict[str, CameraResolution] = {}
        for camera_key, values in declarations.items():
            shapes = {shape for shape, _source in values}
            if len(shapes) != 1:
                detail = ", ".join(
                    f"{source}={width}x{height}"
                    for (height, width), source in values
                )
                raise ValueError(
                    f"camera {camera_key!r} has inconsistent reference resolutions: {detail}"
                )
            height, width = next(iter(shapes))
            self._resolutions[camera_key] = CameraResolution(
                height=height,
                width=width,
                sources=tuple(source for _shape, source in values),
            )

    def get(self, camera_key: str) -> CameraResolution | None:
        return self._resolutions.get(str(camera_key))

    def require(self, camera_key: str) -> CameraResolution:
        key = str(camera_key)
        resolution = self.get(key)
        if resolution is None:
            raise ValueError(
                f"missing camera view {key!r} has no explicit resolution or reference dataset source"
            )
        return resolution


__all__ = ["CameraResolution", "MissingViewResolver"]

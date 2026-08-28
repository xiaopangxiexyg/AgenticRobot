from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_real_image_root() -> Path:
    """Return the repository/provider root used by the default mixed profile.

    The environment override keeps the profile portable to the remote OpenPI
    checkout and to machines that keep the image pool outside the repository.
    The caller still validates the returned root when it constructs a provider.
    """

    configured = os.environ.get("DEXTELEOP_REAL_IMAGE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "random_real_image"
        / "openimages_v7_validation"
        / "images"
    ).resolve()


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("real image background provider requires opencv-python.") from exc
    return cv2


def discover_image_paths(root: str | Path) -> list[Path]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"background image root does not exist: {base}")
    return sorted(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def write_image_manifest(
    image_root: str | Path,
    manifest_path: str | Path,
    *,
    source: str = "local",
    max_images: int = 0,
) -> dict[str, Any]:
    root = Path(image_root).expanduser().resolve()
    paths = discover_image_paths(root)
    if int(max_images) > 0:
        paths = paths[: int(max_images)]
    out = Path(manifest_path).expanduser().resolve()
    try:
        serialized_root = str(root.relative_to(out.parent))
        root_kind = "manifest_relative"
    except ValueError:
        serialized_root = str(root)
        root_kind = "absolute"
    payload = {
        "format": "dexteleop_real_background_manifest_v1",
        "source": str(source),
        "image_root": serialized_root,
        "image_root_kind": root_kind,
        "count": len(paths),
        "images": [str(path.relative_to(root)) for path in paths],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload


def _load_manifest(
    path: str | Path,
    *,
    image_root_override: str | Path | None = None,
) -> tuple[Path, list[Path]]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"real background manifest must contain an object: {manifest_path}")
    if image_root_override is not None:
        root = Path(image_root_override).expanduser().resolve()
    else:
        root = Path(str(payload.get("image_root") or manifest_path.parent)).expanduser()
        if not root.is_absolute():
            root = (manifest_path.parent / root).resolve()
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError(f"real background manifest has no images list: {manifest_path}")
    return root, [(root / str(item)).resolve() for item in images]


@dataclass(frozen=True)
class RealImageBackgroundConfig:
    seed: int = 20260709
    canvas_scale: float = 2.2
    min_scale: float = 0.55
    max_scale: float = 2.4
    min_aspect_stretch: float = 0.50
    max_aspect_stretch: float = 2.0
    min_drift_px_per_frame: float = 0.0
    max_drift_px_per_frame: float = 7.0
    rotation_deg_per_frame: float = 0.04
    zoom_per_frame: float = 0.0006
    color_jitter: float = 0.10
    blur_px: int = 1

    def __post_init__(self) -> None:
        if self.canvas_scale < 1.0:
            raise ValueError("canvas_scale must be >= 1.0")
        if self.min_scale <= 0.0 or self.max_scale < self.min_scale:
            raise ValueError("scale range must be positive and ordered")
        if self.min_aspect_stretch <= 0.0 or self.max_aspect_stretch < self.min_aspect_stretch:
            raise ValueError("aspect stretch range must be positive and ordered")


class LocalImageBackgroundProvider:
    """Deterministic temporal backgrounds from a local image pool."""

    def __init__(
        self,
        *,
        image_root: str | Path | None = None,
        manifest_path: str | Path | None = None,
        config: RealImageBackgroundConfig | None = None,
    ):
        if image_root is None and manifest_path is None:
            raise ValueError("image_root or manifest_path is required")
        self.config = config if config is not None else RealImageBackgroundConfig()
        if manifest_path is not None:
            self.image_root, self.paths = _load_manifest(
                manifest_path,
                image_root_override=image_root,
            )
        else:
            self.image_root = Path(image_root).expanduser().resolve()  # type: ignore[arg-type]
            self.paths = discover_image_paths(self.image_root)
        if not self.paths:
            raise ValueError(f"no background images found under {self.image_root}")

    def stream(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int = 0,
        scene_seed: int | None = None,
        view_seed: int | None = None,
    ) -> "RealImageTemporalBackground":
        default_seed = _sample_seed(
            self.config.seed,
            episode_index=episode_index,
            camera_key=camera_key,
            epoch=epoch,
        )
        path_seed = (
            default_seed
            if scene_seed is None
            else int(self.config.seed) + int(scene_seed)
        )
        stream_seed = (
            default_seed
            if view_seed is None
            else int(self.config.seed) + int(view_seed)
        )
        random = np.random.default_rng(path_seed)
        path = self.paths[int(random.integers(0, len(self.paths)))]
        return RealImageTemporalBackground(
            path=path,
            width=width,
            height=height,
            seed=stream_seed,
            config=self.config,
        )


class RealImageTemporalBackground:
    def __init__(self, *, path: Path, width: int, height: int, seed: int, config: RealImageBackgroundConfig):
        cv2 = _require_cv2()
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.config = config
        self.random = np.random.default_rng(int(seed))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read background image: {path}")
        self.source_scale = float(self.random.uniform(config.min_scale, config.max_scale))
        self.source_stretch_x = float(self.random.uniform(config.min_aspect_stretch, config.max_aspect_stretch))
        self.source_stretch_y = float(self.random.uniform(config.min_aspect_stretch, config.max_aspect_stretch))
        self.canvas = self._make_canvas(image)
        angle = float(self.random.uniform(0.0, 2.0 * np.pi))
        speed = float(self.random.uniform(config.min_drift_px_per_frame, config.max_drift_px_per_frame))
        self.velocity = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32) * speed
        self.origin = np.asarray(
            [
                float(self.random.uniform(0.0, max(1, self.canvas.shape[1] - self.width))),
                float(self.random.uniform(0.0, max(1, self.canvas.shape[0] - self.height))),
            ],
            dtype=np.float32,
        )
        self.rotation0 = float(self.random.uniform(-2.0, 2.0))
        self.rotation_velocity = float(config.rotation_deg_per_frame) * float(self.random.uniform(-1.0, 1.0))
        self.zoom0 = float(self.random.uniform(0.96, 1.06))
        self.zoom_velocity = float(config.zoom_per_frame) * float(self.random.uniform(-1.0, 1.0))

    def _make_canvas(self, image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        target_w = max(self.width + 8, int(round(self.width * max(1.2, self.config.canvas_scale))))
        target_h = max(self.height + 8, int(round(self.height * max(1.2, self.config.canvas_scale))))
        src_h, src_w = image.shape[:2]
        cover_scale = max(target_w / max(1, src_w), target_h / max(1, src_h))
        width = max(2, int(round(src_w * cover_scale * self.source_scale * self.source_stretch_x)))
        height = max(2, int(round(src_h * cover_scale * self.source_scale * self.source_stretch_y)))
        interpolation = cv2.INTER_AREA if width < src_w or height < src_h else cv2.INTER_LINEAR
        resized = cv2.resize(image, (width, height), interpolation=interpolation)
        pad_x = max(0, target_w - resized.shape[1])
        pad_y = max(0, target_h - resized.shape[0])
        if pad_x or pad_y:
            resized = cv2.copyMakeBorder(
                resized,
                pad_y // 2,
                pad_y - pad_y // 2,
                pad_x // 2,
                pad_x - pad_x // 2,
                borderType=cv2.BORDER_REFLECT_101,
            )
        max_x = max(0, resized.shape[1] - target_w)
        max_y = max(0, resized.shape[0] - target_h)
        x = int(self.random.integers(0, max_x + 1)) if max_x > 0 else 0
        y = int(self.random.integers(0, max_y + 1)) if max_y > 0 else 0
        canvas = resized[y : y + target_h, x : x + target_w].copy()
        if float(self.config.color_jitter) > 0.0:
            scale_bgr = self.random.uniform(1.0 - self.config.color_jitter, 1.0 + self.config.color_jitter, size=(1, 1, 3))
            bias = self.random.uniform(-12.0, 12.0, size=(1, 1, 3))
            canvas = np.clip(canvas.astype(np.float32) * scale_bgr + bias, 0, 255).astype(np.uint8)
        if int(self.config.blur_px) > 0:
            radius = int(self.config.blur_px)
            canvas = cv2.GaussianBlur(canvas, (radius * 2 + 1, radius * 2 + 1), 0.0)
        return canvas

    def frame(self, frame_index: int) -> np.ndarray:
        cv2 = _require_cv2()
        t = float(max(0, int(frame_index)))
        canvas_h, canvas_w = self.canvas.shape[:2]
        max_x = max(1.0, float(canvas_w - self.width))
        max_y = max(1.0, float(canvas_h - self.height))
        offset = self.origin + self.velocity * t
        x = int(np.floor(float(offset[0] % max_x)))
        y = int(np.floor(float(offset[1] % max_y)))
        padded = cv2.copyMakeBorder(
            self.canvas,
            self.height,
            self.height,
            self.width,
            self.width,
            borderType=cv2.BORDER_REFLECT,
        )
        crop = padded[y + self.height : y + self.height + self.height, x + self.width : x + self.width + self.width].copy()
        angle = self.rotation0 + self.rotation_velocity * t
        zoom = np.clip(self.zoom0 + self.zoom_velocity * t, 0.92, 1.10)
        matrix = cv2.getRotationMatrix2D((self.width * 0.5, self.height * 0.5), angle, zoom)
        return cv2.warpAffine(crop, matrix, (self.width, self.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _sample_seed(seed: int, *, episode_index: int, camera_key: str, epoch: int) -> int:
    camera_hash = sum((index + 1) * ord(ch) for index, ch in enumerate(str(camera_key)))
    return int(seed) + int(episode_index) * 100003 + camera_hash + int(epoch) * 10000019

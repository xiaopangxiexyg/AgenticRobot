from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

from utils.random_background import RandomBackgroundAugmentConfig


ImageLayout = Literal["bgr_hwc_uint8", "rgb_hwc_uint8", "rgb_chw_float32"]


@dataclass(frozen=True)
class MaskConfig:
    final_mask_root: Path
    mask_role: Literal["keep", "replace"] = "keep"
    manifest_name: str = "manifest.json"


@dataclass(frozen=True)
class VideoConfig:
    reader_cache_size: int = 16
    mask_cache_size: int = 64
    output_layout: ImageLayout = "bgr_hwc_uint8"


@dataclass(frozen=True)
class MissingViewConfig:
    """Model-selected sources for synthesizing absent camera views.

    Resolutions always use ``(height, width)`` ordering.  Reference datasets
    are consulted only for camera metadata; their frames are never sampled as
    content for the missing view.
    """

    enabled: bool = False
    reference_dataset_roots: tuple[Path, ...] = ()
    explicit_resolutions: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        roots = tuple(Path(value).expanduser().resolve() for value in self.reference_dataset_roots)
        resolutions: dict[str, tuple[int, int]] = {}
        for raw_key, raw_shape in dict(self.explicit_resolutions).items():
            camera_key = str(raw_key).strip()
            if not camera_key:
                raise ValueError("MissingViewConfig explicit camera keys must be non-empty")
            if not isinstance(raw_shape, (tuple, list)) or len(raw_shape) != 2:
                raise ValueError(
                    f"MissingViewConfig resolution for {camera_key!r} must be (height, width)"
                )
            height, width = int(raw_shape[0]), int(raw_shape[1])
            if height < 1 or width < 1:
                raise ValueError(
                    f"MissingViewConfig resolution for {camera_key!r} must be positive"
                )
            resolutions[camera_key] = (height, width)
        object.__setattr__(self, "reference_dataset_roots", roots)
        object.__setattr__(self, "explicit_resolutions", resolutions)


@dataclass(frozen=True)
class DatasetInterfaceConfig:
    dataset_root: Path
    mask: MaskConfig
    video: VideoConfig = field(default_factory=VideoConfig)
    augmentation: RandomBackgroundAugmentConfig = field(default_factory=RandomBackgroundAugmentConfig)
    missing_views: MissingViewConfig = field(default_factory=MissingViewConfig)

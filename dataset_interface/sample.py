from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameRequest:
    episode_index: int
    camera_key: str
    frame_index: int
    epoch: int = 0
    occlusion_ratio: float = 0.0
    occlusion_scope_anchor: int | None = None
    forced_missing_view: bool = False


@dataclass(frozen=True)
class MaskedFrameSample:
    request: FrameRequest
    image: np.ndarray
    mask: np.ndarray
    augmented: np.ndarray
    video_path: Path | None
    mask_manifest_path: Path | None
    metadata: dict[str, Any]

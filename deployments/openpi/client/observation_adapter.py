from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


DEFAULT_IMAGE_KEY_MAP: dict[str, str] = {
    "front": "observation.images.front",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


@dataclass(frozen=True)
class OpenPIObservationAdapterConfig:
    prompt: str | None = None
    image_key_map: Mapping[str, str] | None = None
    required_image_keys: tuple[str, ...] = ("front",)
    include_debug_metadata: bool = True

    def __post_init__(self) -> None:
        image_key_map = DEFAULT_IMAGE_KEY_MAP if self.image_key_map is None else dict(self.image_key_map)
        object.__setattr__(self, "image_key_map", dict(image_key_map))
        object.__setattr__(self, "required_image_keys", tuple(str(key) for key in self.required_image_keys))


def deployment_observation_to_openpi(
    observation: Any,
    config: OpenPIObservationAdapterConfig | None = None,
) -> dict[str, Any]:
    config = OpenPIObservationAdapterConfig() if config is None else config
    images = dict(getattr(observation, "images", {}) or {})
    output: dict[str, Any] = {}

    assert config.image_key_map is not None
    for source_key, openpi_key in config.image_key_map.items():
        if source_key not in images:
            if source_key in set(config.required_image_keys):
                raise ValueError(f"Missing required OpenPI image key: {source_key!r}.")
            continue
        output[str(openpi_key)] = _as_uint8_hwc(images[source_key], key=source_key)

    state = getattr(observation, "state", None)
    if state is None:
        raise ValueError("Deployment observation is missing state.")
    output["state"] = np.asarray(state, dtype=np.float32).reshape(-1)

    prompt = _resolve_prompt(observation, config)
    if prompt:
        output["prompt"] = prompt

    if config.include_debug_metadata:
        output["dexteleop"] = {
            "timestamp": float(getattr(observation, "timestamp", 0.0) or 0.0),
            "step_index": int(getattr(observation, "step_index", 0) or 0),
            "active_sides": list(getattr(observation, "active_sides", ()) or ()),
            "arms_enabled": bool(getattr(observation, "arms_enabled", False)),
            "hands_enabled": bool(getattr(observation, "hands_enabled", False)),
            "faults": list(getattr(observation, "faults", ()) or ()),
        }
    return output


def _resolve_prompt(observation: Any, config: OpenPIObservationAdapterConfig) -> str | None:
    if config.prompt:
        return str(config.prompt)
    metadata = getattr(observation, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        for key in ("prompt", "instruction", "task_instruction"):
            value = metadata.get(key)
            if value:
                return str(value)
        task = metadata.get("task")
        if isinstance(task, Mapping):
            value = task.get("instruction")
            if value:
                return str(value)
    return None


def _as_uint8_hwc(value: Any, *, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Image {key!r} must be 3D, got shape {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Image {key!r} must have 3 color channels, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        finite = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(finite * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


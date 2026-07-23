from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from robot_motion.deploy.normalization import (
    NormalizationContract,
    NormalizationContractError,
    normalization_contract_hash,
)


DEFAULT_IMAGE_KEY_MAP: dict[str, str] = {
    "front": "observation.images.front",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


@dataclass(frozen=True)
class OpenPIObservationAdapterConfig:
    prompt: str | None = None
    image_key_map: Mapping[str, str] | None = None
    required_image_keys: tuple[str, ...] | None = None
    include_debug_metadata: bool = True
    state_key: str = "state"
    state_discrete_key: str = "state_discrete"
    # Optional explicit numeric contract.  ``None`` preserves the historical
    # payload shape; when present it is carried in debug metadata so the server
    # can bind the same owner/hash.  This adapter does not infer or execute a
    # non-identity transform from a legacy ``normalized`` label.
    normalization_contract: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        image_key_map = DEFAULT_IMAGE_KEY_MAP if self.image_key_map is None else dict(self.image_key_map)
        normalized_map: dict[str, str] = {}
        for source_key, model_input_key in image_key_map.items():
            source = str(source_key).strip()
            destination = str(model_input_key).strip()
            if not source or not destination:
                raise ValueError("OpenPI image_key_map keys and values must be non-empty.")
            normalized_map[source] = destination
        if len(set(normalized_map.values())) != len(normalized_map):
            raise ValueError("OpenPI image_key_map model input keys must be unique.")
        requested_required = self.required_image_keys
        required_image_keys = (
            ("front",)
            if requested_required is None and self.image_key_map is None
            else tuple(normalized_map)
            if requested_required is None
            else tuple(str(key).strip() for key in requested_required)
        )
        if any(not key for key in required_image_keys):
            raise ValueError("OpenPI required_image_keys must be non-empty.")
        unknown_required = set(required_image_keys).difference(normalized_map)
        if unknown_required:
            raise ValueError(
                "OpenPI required_image_keys are absent from image_key_map: "
                f"{sorted(unknown_required)}."
            )
        object.__setattr__(self, "image_key_map", normalized_map)
        object.__setattr__(self, "required_image_keys", required_image_keys)
        object.__setattr__(self, "state_key", str(self.state_key))
        object.__setattr__(self, "state_discrete_key", str(self.state_discrete_key))
        if self.normalization_contract is not None:
            try:
                parsed = NormalizationContract.from_optional_mapping(
                    self.normalization_contract,
                    allow_legacy=True,
                )
            except NormalizationContractError as exc:
                raise ValueError(str(exc)) from exc
            if parsed is not None:
                if parsed.has_client_owned_transform:
                    raise ValueError(
                        "OpenPI observation adapter does not execute client-owned "
                        "normalization; provide a model-specific executor or declare "
                        "the server as owner."
                    )
                object.__setattr__(self, "normalization_contract", parsed.to_dict())
            else:
                object.__setattr__(
                    self,
                    "normalization_contract",
                    {str(key): value for key, value in self.normalization_contract.items()},
                )

    @property
    def normalization_hash(self) -> str | None:
        if self.normalization_contract is None:
            return None
        return normalization_contract_hash(self.normalization_contract)


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
    output[config.state_key] = np.asarray(state, dtype=np.float32).reshape(-1)
    state_discrete = getattr(observation, "state_discrete", None)
    if state_discrete is not None:
        output[config.state_discrete_key] = _as_int64_tokens(
            state_discrete,
            key=config.state_discrete_key,
        )

    prompt = _resolve_prompt(observation, config)
    if prompt:
        output["prompt"] = prompt

    if config.include_debug_metadata:
        debug_metadata = {
            "timestamp": float(getattr(observation, "timestamp", 0.0) or 0.0),
            "step_index": int(getattr(observation, "step_index", 0) or 0),
            "active_sides": list(getattr(observation, "active_sides", ()) or ()),
            "arms_enabled": bool(getattr(observation, "arms_enabled", False)),
            "hands_enabled": bool(getattr(observation, "hands_enabled", False)),
            "policy_io_layout_hash": getattr(observation, "policy_io_layout_hash", None),
        }
        if config.normalization_contract is not None:
            parsed = NormalizationContract.from_optional_mapping(
                config.normalization_contract,
                allow_legacy=True,
            )
            debug_metadata.update(
                {
                    "normalization_hash": config.normalization_hash,
                    "normalization_contract_status": (
                        "explicit" if parsed is not None else "legacy_opaque"
                    ),
                }
            )
        output["dexteleop"] = debug_metadata
    return output


def _resolve_prompt(observation: Any, config: OpenPIObservationAdapterConfig) -> str | None:
    if config.prompt:
        return str(config.prompt)
    for key in ("prompt", "instruction", "task_instruction"):
        value = getattr(observation, key, None)
        if value:
            return str(value)
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


def _as_int64_tokens(value: Any, *, key: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{key!r} must be a 1D token vector, got shape {raw.shape}.")
    tokens = np.empty(raw.shape, dtype=np.int64)
    for index, item in enumerate(raw):
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{key!r}[{index}] must be an integer token, not a boolean.")
        try:
            token = int(item)
            if float(item) != float(token):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key!r}[{index}] must be an exact integer token.") from exc
        tokens[index] = token
    return tokens

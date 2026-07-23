"""Pure processed-sample contract for the external ``vla-scratch`` stack.

The contract mirrors the public keys consumed by ``ServePolicy.infer`` while
remaining independent from vla-scratch, Torch, ZMQ, training code and robot
runtime code.  Dataset and deployment adapters can therefore build the same
model-facing observation without making either repository import the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .canonical_encoding import canonical_content_hash


VLA_SCRATCH_PROCESSED_SAMPLE_V1 = "vla_scratch_processed_sample_v1"
VLA_SCRATCH_IMAGES_KEY = "observation.images"
VLA_SCRATCH_IMAGE_MASKS_KEY = "observation.image_masks"
VLA_SCRATCH_STATE_KEY = "observation.state"
VLA_SCRATCH_TASK_KEY = "task"
VLA_SCRATCH_ACTIONS_KEY = "action_chunk.actions"
VLA_SCRATCH_SAMPLE_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.vla_scratch_processed_sample_spec.v1"
)
VLA_SCRATCH_OBSERVATION_HASH_DOMAIN_V1 = (
    "dexteleop.vla_scratch_processed_observation.v1"
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized <= 0 or normalized != value:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


@dataclass(frozen=True)
class VlaScratchProcessedSampleSpec:
    """Exact common input/output shape contract for one loaded VLA policy."""

    source_revision: str
    camera_source_keys: tuple[str, ...]
    state_history: int
    state_dim: int
    action_horizon: int
    action_dim: int
    state_representation_id: str
    action_representation_id: str
    schema: str = VLA_SCRATCH_PROCESSED_SAMPLE_V1

    def __post_init__(self) -> None:
        if self.schema != VLA_SCRATCH_PROCESSED_SAMPLE_V1:
            raise ValueError(f"Unsupported vla-scratch sample schema: {self.schema!r}.")
        for name in (
            "source_revision",
            "state_representation_id",
            "action_representation_id",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        cameras = tuple(
            _nonempty(value, name="camera_source_key")
            for value in self.camera_source_keys
        )
        if not cameras:
            raise ValueError("camera_source_keys must not be empty.")
        if len(set(cameras)) != len(cameras):
            raise ValueError("camera_source_keys must not contain duplicates.")
        object.__setattr__(self, "camera_source_keys", cameras)
        for name in (
            "state_history",
            "state_dim",
            "action_horizon",
            "action_dim",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_revision": self.source_revision,
            "camera_source_keys": list(self.camera_source_keys),
            "state_history": self.state_history,
            "state_dim": self.state_dim,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "state_representation_id": self.state_representation_id,
            "action_representation_id": self.action_representation_id,
            "wire": {
                "images_key": VLA_SCRATCH_IMAGES_KEY,
                "images_layout": "NCHW",
                "images_dtype": "uint8",
                "image_masks_key": VLA_SCRATCH_IMAGE_MASKS_KEY,
                "image_masks_shape": [len(self.camera_source_keys), 1],
                "state_key": VLA_SCRATCH_STATE_KEY,
                "state_dtype": "float32",
                "task_key": VLA_SCRATCH_TASK_KEY,
                "actions_key": VLA_SCRATCH_ACTIONS_KEY,
                "actions_dtype": "float32",
            },
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=VLA_SCRATCH_SAMPLE_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class VlaScratchProcessedObservation:
    """Validated serving observation matching vla-scratch processed keys."""

    spec: VlaScratchProcessedSampleSpec
    images: np.ndarray
    image_masks: np.ndarray
    state: np.ndarray
    task: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, VlaScratchProcessedSampleSpec):
            raise TypeError("spec must be VlaScratchProcessedSampleSpec.")
        images = np.asarray(self.images)
        expected_cameras = len(self.spec.camera_source_keys)
        if images.dtype != np.uint8:
            raise TypeError("vla-scratch processed images must be uint8.")
        if (
            images.ndim != 4
            or images.shape[0] != expected_cameras
            or images.shape[1] != 3
            or images.shape[2] <= 0
            or images.shape[3] <= 0
        ):
            raise ValueError(
                "vla-scratch processed images must have shape "
                f"[{expected_cameras},3,H,W], got {images.shape}."
            )
        masks = np.asarray(self.image_masks)
        if masks.dtype != np.bool_ or masks.shape != (expected_cameras, 1):
            raise ValueError(
                "vla-scratch image masks must be bool with shape "
                f"[{expected_cameras},1]."
            )
        state = np.asarray(self.state)
        expected_state_shape = (self.spec.state_history, self.spec.state_dim)
        if state.dtype != np.float32 or state.shape != expected_state_shape:
            raise ValueError(
                "vla-scratch state must be float32 with shape "
                f"{expected_state_shape}, got {state.dtype} {state.shape}."
            )
        if np.any(~np.isfinite(state)):
            raise ValueError("vla-scratch state must be finite.")
        task = _nonempty(self.task, name="task")

        images = np.ascontiguousarray(images).copy()
        masks = np.ascontiguousarray(masks).copy()
        state = np.ascontiguousarray(state).copy()
        images.setflags(write=False)
        masks.setflags(write=False)
        state.setflags(write=False)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "image_masks", masks)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "task", task)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "sample_spec_hash": self.spec.semantic_content_hash,
                "images": self.images,
                "image_masks": self.image_masks,
                "state": self.state,
                "task": self.task,
            },
            domain=VLA_SCRATCH_OBSERVATION_HASH_DOMAIN_V1,
        )

    def to_wire_payload(self) -> dict[str, Any]:
        """Return detached arrays for the existing vla-scratch client."""

        return {
            VLA_SCRATCH_IMAGES_KEY: self.images.copy(),
            VLA_SCRATCH_IMAGE_MASKS_KEY: self.image_masks.copy(),
            VLA_SCRATCH_STATE_KEY: self.state.copy(),
            VLA_SCRATCH_TASK_KEY: self.task,
        }


def _as_uint8_chw(value: Any, *, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Image {key!r} must be rank 3, got {image.shape}.")
    if image.shape[-1] == 3:
        image = np.moveaxis(image, -1, 0)
    elif image.shape[0] != 3:
        raise ValueError(
            f"Image {key!r} must be RGB HWC or CHW, got {image.shape}."
        )
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if np.any(~np.isfinite(image)) or np.any(image < 0.0) or np.any(image > 1.0):
            raise ValueError(
                f"Floating image {key!r} must be finite and normalized to [0,1]."
            )
        return np.ascontiguousarray(np.rint(image * 255.0).astype(np.uint8))
    raise TypeError(f"Image {key!r} must be uint8 or normalized floating point.")


def build_vla_scratch_processed_observation(
    *,
    spec: VlaScratchProcessedSampleSpec,
    images_by_source: Mapping[str, Any],
    state: Any,
    task: str,
) -> VlaScratchProcessedObservation:
    """Build the shared training/serving observation with exact ordering."""

    if not isinstance(spec, VlaScratchProcessedSampleSpec):
        raise TypeError("spec must be VlaScratchProcessedSampleSpec.")
    if not isinstance(images_by_source, Mapping):
        raise TypeError("images_by_source must be a mapping.")
    missing = tuple(
        key for key in spec.camera_source_keys if key not in images_by_source
    )
    if missing:
        raise ValueError(f"Missing required vla-scratch camera sources: {missing}.")
    ordered = tuple(
        _as_uint8_chw(images_by_source[key], key=key)
        for key in spec.camera_source_keys
    )
    spatial_shapes = {image.shape[1:] for image in ordered}
    if len(spatial_shapes) != 1:
        raise ValueError(
            "vla-scratch processed images must share one spatial resolution."
        )
    images = np.stack(ordered, axis=0)
    masks = np.ones((len(ordered), 1), dtype=np.bool_)

    state_array = np.asarray(state, dtype=np.float32)
    expected_shape = (spec.state_history, spec.state_dim)
    if state_array.ndim == 1 and state_array.size == spec.state_history * spec.state_dim:
        state_array = state_array.reshape(expected_shape)
    if state_array.shape != expected_shape:
        raise ValueError(
            f"vla-scratch state must have shape {expected_shape}, "
            f"got {state_array.shape}."
        )
    return VlaScratchProcessedObservation(
        spec=spec,
        images=images,
        image_masks=masks,
        state=np.ascontiguousarray(state_array, dtype=np.float32),
        task=task,
    )


def validate_vla_scratch_action_chunk(
    value: Any,
    *,
    spec: VlaScratchProcessedSampleSpec,
) -> np.ndarray:
    """Validate an action target or server result against the same binding."""

    actions = np.asarray(value)
    expected_shape = (spec.action_horizon, spec.action_dim)
    if not np.issubdtype(actions.dtype, np.floating):
        raise TypeError("vla-scratch actions must use a floating dtype.")
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != expected_shape:
        raise ValueError(
            f"vla-scratch actions must have shape {expected_shape}, "
            f"got {actions.shape}."
        )
    if np.any(~np.isfinite(actions)):
        raise ValueError("vla-scratch actions must be finite.")
    result = np.ascontiguousarray(actions).copy()
    result.setflags(write=False)
    return result


__all__ = [
    "VLA_SCRATCH_ACTIONS_KEY",
    "VLA_SCRATCH_IMAGE_MASKS_KEY",
    "VLA_SCRATCH_IMAGES_KEY",
    "VLA_SCRATCH_PROCESSED_SAMPLE_V1",
    "VLA_SCRATCH_STATE_KEY",
    "VLA_SCRATCH_TASK_KEY",
    "VlaScratchProcessedObservation",
    "VlaScratchProcessedSampleSpec",
    "build_vla_scratch_processed_observation",
    "validate_vla_scratch_action_chunk",
]

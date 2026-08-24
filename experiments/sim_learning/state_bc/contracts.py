"""Contracts for the first state-only behavior-cloning baseline.

The real FormalCollection low-dimensional contract is deliberately kept at
the edge of this experiment.  This module owns no runtime, simulator, device,
or dataset persistence authority.  It only validates the 16-dimensional
state/action vectors and records the identity needed to reproduce an offline
fit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


STATE_BC_SCHEMA_V1 = "dexteleop.state_bc_experiment.v1"
STATE_BC_MODEL_SCHEMA_V1 = "dexteleop.state_bc_model.v1"
STATE_BC_LAYOUT_HASH_V1 = (
    "sha256:2567572f0f9170281b3e775ed13ecba903fbdeb199fd88a5bb1b7b1f4e96db16"
)
# The checked-in runtime layout currently serializes the same v2.2 channels
# with a newer derived-layout hash.  Treat it as an explicit compatibility
# alias, never as an unvalidated wildcard, so an artifact records which source
# metadata it consumed while the simulator binds the current runtime layout.
STATE_BC_RUNTIME_LAYOUT_HASH_V1 = (
    "sha256:52e4bc42ad0a0e562c357e41da1ee2cbd827b4dc98643ffb73fe7acfeafc66ab"
)
STATE_BC_ACCEPTED_LAYOUT_HASHES = frozenset(
    {STATE_BC_LAYOUT_HASH_V1, STATE_BC_RUNTIME_LAYOUT_HASH_V1}
)
STATE_BC_STATE_DIM = 16
STATE_BC_ACTION_DIM = 16
STATE_BC_POSE_FRAME = "desk_absolute"
STATE_BC_POSE_REPRESENTATION = "xyz_quat_xyzw"
STATE_BC_ACTION_MODE = "absolute"
STATE_BC_GRIPPER_SEMANTICS = "normalized_0_1"

STATE_BC_POSE_SLICES: dict[str, slice] = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}
STATE_BC_GRIPPER_SLICES: dict[str, slice] = {
    "left": slice(14, 15),
    "right": slice(15, 16),
}


def _required_text(value: Any, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite_vector(value: Any, *, name: str, dim: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector.") from exc
    if vector.shape != (int(dim),):
        raise ValueError(f"{name} must have shape ({dim},), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector.astype(np.float32, copy=True)


def validate_state_vector(value: Any, *, name: str = "state") -> np.ndarray:
    """Validate one real/sim model input without applying normalization."""

    return _finite_vector(value, name=name, dim=STATE_BC_STATE_DIM)


def validate_action_vector(value: Any, *, name: str = "action") -> np.ndarray:
    """Validate one absolute 16D policy action."""

    return _finite_vector(value, name=name, dim=STATE_BC_ACTION_DIM)


def content_hash(value: Any) -> str:
    """Return a stable hash for JSON-compatible experiment metadata."""

    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class StateBCSelector:
    """A reproducible set of FormalCollection source directories.

    A selector names all source directories explicitly.  This is important
    for the 2026-08-07 logical day, which is split into two physical folders,
    and prevents a trainer from silently scanning unrelated dates.
    """

    selector_id: str
    source_dirs: tuple[str, ...]
    require_collection_integrity_valid: bool = True
    min_episode_frames: int = 2
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        selector_id = _required_text(self.selector_id, name="selector_id")
        source_dirs = tuple(_required_text(value, name="source_dir") for value in self.source_dirs)
        if not source_dirs:
            raise ValueError("source_dirs must contain at least one directory.")
        if len(set(source_dirs)) != len(source_dirs):
            raise ValueError("source_dirs must not contain duplicates.")
        min_frames = int(self.min_episode_frames)
        if min_frames <= 0:
            raise ValueError("min_episode_frames must be positive.")
        object.__setattr__(self, "selector_id", selector_id)
        object.__setattr__(self, "source_dirs", source_dirs)
        object.__setattr__(self, "require_collection_integrity_valid", bool(self.require_collection_integrity_valid))
        object.__setattr__(self, "min_episode_frames", min_frames)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def semantic_content_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_BC_SCHEMA_V1,
            "selector_id": self.selector_id,
            "source_dirs": list(self.source_dirs),
            "require_collection_integrity_valid": self.require_collection_integrity_valid,
            "min_episode_frames": self.min_episode_frames,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StateBCNormalization:
    """Feature normalization owned by the offline learner."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = _finite_vector(self.mean, name="normalization.mean", dim=STATE_BC_STATE_DIM)
        scale = _finite_vector(self.scale, name="normalization.scale", dim=STATE_BC_STATE_DIM)
        if np.any(scale <= 0.0):
            raise ValueError("normalization.scale must be strictly positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def transform(self, state: Any) -> np.ndarray:
        value = validate_state_vector(state)
        return ((value - self.mean) / self.scale).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


@dataclass(frozen=True)
class StateBCArtifactMetadata:
    """Serialized identity shared by the trainer and the simulator adapter."""

    selector_id: str
    train_episode_count: int
    test_episode_count: int
    train_frame_count: int
    test_frame_count: int
    model_type: str = "linear_ridge_v1"
    artifact_schema: str = STATE_BC_MODEL_SCHEMA_V1
    layout_hash: str = STATE_BC_LAYOUT_HASH_V1
    state_dim: int = STATE_BC_STATE_DIM
    action_dim: int = STATE_BC_ACTION_DIM
    pose_frame: str = STATE_BC_POSE_FRAME
    pose_representation: str = STATE_BC_POSE_REPRESENTATION
    action_mode: str = STATE_BC_ACTION_MODE
    gripper_semantics: str = STATE_BC_GRIPPER_SEMANTICS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        selector_id = _required_text(self.selector_id, name="selector_id")
        model_type = _required_text(self.model_type, name="model_type")
        if str(self.artifact_schema) != STATE_BC_MODEL_SCHEMA_V1:
            raise ValueError("unsupported state BC artifact schema.")
        if str(self.layout_hash) not in STATE_BC_ACCEPTED_LAYOUT_HASHES:
            raise ValueError("state BC artifact must bind an accepted v2.2 layout hash.")
        if int(self.state_dim) != STATE_BC_STATE_DIM or int(self.action_dim) != STATE_BC_ACTION_DIM:
            raise ValueError("state BC artifact dimensions must both be 16.")
        for name in ("train_episode_count", "test_episode_count", "train_frame_count", "test_frame_count"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if str(self.pose_frame) != STATE_BC_POSE_FRAME:
            raise ValueError("state BC actions must be desk_absolute.")
        if str(self.pose_representation) != STATE_BC_POSE_REPRESENTATION:
            raise ValueError("state BC poses must use xyz_quat_xyzw.")
        if str(self.action_mode) != STATE_BC_ACTION_MODE:
            raise ValueError("state BC actions must be absolute targets.")
        if str(self.gripper_semantics) != STATE_BC_GRIPPER_SEMANTICS:
            raise ValueError("state BC grippers must use normalized_0_1 semantics.")
        object.__setattr__(self, "selector_id", selector_id)
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.artifact_schema,
            "selector_id": self.selector_id,
            "model_type": self.model_type,
            "layout_hash": self.layout_hash,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "pose_frame": self.pose_frame,
            "pose_representation": self.pose_representation,
            "action_mode": self.action_mode,
            "gripper_semantics": self.gripper_semantics,
            "train_episode_count": self.train_episode_count,
            "test_episode_count": self.test_episode_count,
            "train_frame_count": self.train_frame_count,
            "test_frame_count": self.test_frame_count,
            "metadata": dict(self.metadata),
        }


def normalize_episode_ids(values: Sequence[Any]) -> tuple[str, ...]:
    ids = tuple(_required_text(value, name="episode_id") for value in values)
    if not ids:
        raise ValueError("episode_ids must not be empty.")
    return ids


__all__ = [
    "STATE_BC_ACTION_DIM",
    "STATE_BC_ACTION_MODE",
    "STATE_BC_ACCEPTED_LAYOUT_HASHES",
    "STATE_BC_GRIPPER_SEMANTICS",
    "STATE_BC_LAYOUT_HASH_V1",
    "STATE_BC_RUNTIME_LAYOUT_HASH_V1",
    "STATE_BC_MODEL_SCHEMA_V1",
    "STATE_BC_POSE_FRAME",
    "STATE_BC_POSE_REPRESENTATION",
    "STATE_BC_POSE_SLICES",
    "STATE_BC_SCHEMA_V1",
    "STATE_BC_STATE_DIM",
    "StateBCArtifactMetadata",
    "StateBCNormalization",
    "StateBCSelector",
    "content_hash",
    "normalize_episode_ids",
    "validate_action_vector",
    "validate_state_vector",
]

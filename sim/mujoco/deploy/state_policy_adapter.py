"""State-only policy adapter for the existing MuJoCo deployment target.

This is a model and representation bridge, not a second runtime.  It loads a
self-contained state-BC artifact (or accepts an injected test model), projects
one 16D prediction through the existing v2.2/q6 compatibility adapter, and
returns ``NormalizedAction`` values for the already-authoritative safety,
IK, and dispatch path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract
from robot_motion.deploy import (
    DeploymentObservation,
    NormalizedAction,
    POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
    POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
    V22DeploymentAdapter,
    V22PolicyActionChunk,
    convert_policy_absolute_vector_to_runtime_raw,
)

from .state_policy_observation import (
    STATE_BC_ACTION_DIM,
    STATE_BC_ACCEPTED_LAYOUT_HASHES,
    STATE_BC_LAYOUT_HASH_V1,
    STATE_BC_STATE_DIM,
    StatePolicyObservation,
    StatePolicyObservationAdapter,
    StatePolicyObservationConfig,
)


STATE_BC_MODEL_SCHEMA_V1 = "dexteleop.state_bc_model.v1"


class StatePolicyModel(Protocol):
    """Minimal model protocol accepted by the simulator adapter."""

    def predict(self, state: Any) -> Any:
        """Return one finite 16D absolute state-policy action."""


def _finite_vector(value: Any, *, name: str, dim: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector.") from exc
    if vector.shape != (dim,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite with shape ({dim},), got {vector.shape}.")
    return vector.astype(np.float32, copy=True)


def _normalize_pose_quaternion(action: np.ndarray, *, name: str) -> np.ndarray:
    value = action.astype(np.float32, copy=True)
    for side, start in (("left", 0), ("right", 7)):
        quaternion = value[start + 3 : start + 7]
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f"{name}.{side} quaternion norm is zero or non-finite.")
        value[start + 3 : start + 7] = quaternion / np.float32(norm)
    return value


class _NpzStatePolicy:
    """Dependency-light reader for the experiment's non-pickle artifact."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        with np.load(self.path, allow_pickle=False) as payload:
            required = {"weights", "bias", "input_mean", "input_scale", "metadata_json"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"state BC artifact is missing arrays: {missing}.")
            raw_metadata = payload["metadata_json"]
            text = str(raw_metadata.item() if raw_metadata.ndim == 0 else raw_metadata)
            metadata = json.loads(text)
            if not isinstance(metadata, Mapping):
                raise ValueError("state BC artifact metadata_json must be an object.")
            weights = np.asarray(payload["weights"], dtype=np.float32)
            bias = np.asarray(payload["bias"], dtype=np.float32).reshape(-1)
            mean = np.asarray(payload["input_mean"], dtype=np.float32).reshape(-1)
            scale = np.asarray(payload["input_scale"], dtype=np.float32).reshape(-1)
        if weights.shape != (STATE_BC_STATE_DIM, STATE_BC_ACTION_DIM) or not np.all(np.isfinite(weights)):
            raise ValueError("state BC artifact weights must have shape (16, 16).")
        if bias.shape != (STATE_BC_ACTION_DIM,) or not np.all(np.isfinite(bias)):
            raise ValueError("state BC artifact bias must have shape (16,).")
        if mean.shape != (STATE_BC_STATE_DIM,) or not np.all(np.isfinite(mean)):
            raise ValueError("state BC artifact input_mean must have shape (16,).")
        if scale.shape != (STATE_BC_STATE_DIM,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("state BC artifact input_scale must be positive with shape (16,).")
        if str(metadata.get("schema")) != STATE_BC_MODEL_SCHEMA_V1:
            raise ValueError("unsupported state BC artifact schema.")
        if str(metadata.get("layout_hash")) not in STATE_BC_ACCEPTED_LAYOUT_HASHES:
            raise ValueError("state BC artifact is not bound to an accepted v2.2 layout.")
        if int(metadata.get("state_dim", -1)) != STATE_BC_STATE_DIM or int(metadata.get("action_dim", -1)) != STATE_BC_ACTION_DIM:
            raise ValueError("state BC artifact dimensions must both be 16.")
        declared_hash = metadata.get("artifact_hash")
        hash_metadata = dict(metadata)
        hash_metadata.pop("artifact_hash", None)
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                hash_metadata,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(np.ascontiguousarray(weights).tobytes())
        digest.update(np.ascontiguousarray(bias).tobytes())
        digest.update(np.ascontiguousarray(mean).tobytes())
        digest.update(np.ascontiguousarray(scale).tobytes())
        computed_hash = f"sha256:{digest.hexdigest()}"
        if declared_hash is not None and str(declared_hash) != computed_hash:
            raise ValueError("state BC artifact hash does not match its contents.")
        self.weights = weights
        self.bias = bias
        self.mean = mean
        self.scale = scale
        self.metadata = dict(metadata)
        self.artifact_hash = computed_hash

    def predict(self, state: Any) -> np.ndarray:
        value = _finite_vector(state, name="state", dim=STATE_BC_STATE_DIM)
        normalized = (value - self.mean) / self.scale
        return _finite_vector(
            normalized @ self.weights + self.bias,
            name="predicted_action",
            dim=STATE_BC_ACTION_DIM,
        )


@dataclass(frozen=True)
class StatePolicyAdapterConfig:
    """Model-frame and output binding for one state-policy adapter."""

    active_sides: tuple[str, ...] = ("left", "right")
    policy_absolute_frame: str = POLICY_ABSOLUTE_FRAME_CANONICAL_DESK
    runtime_pose_frame_contract: PoseFrameContract | Mapping[str, Any] | None = None
    expected_layout_hash: str = STATE_BC_LAYOUT_HASH_V1
    source: str = "state_bc_policy"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observation_config = StatePolicyObservationConfig(
            active_sides=self.active_sides,
            policy_absolute_frame=self.policy_absolute_frame,
            runtime_pose_frame_contract=self.runtime_pose_frame_contract,
            expected_layout_hash=self.expected_layout_hash,
        )
        object.__setattr__(self, "active_sides", observation_config.active_sides)
        object.__setattr__(self, "policy_absolute_frame", observation_config.policy_absolute_frame)
        object.__setattr__(self, "expected_layout_hash", observation_config.expected_layout_hash)
        source = str(self.source).strip()
        if not source:
            raise ValueError("source must be non-empty.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class StatePolicyPrediction:
    """One model prediction ready for the existing deployment action ingress."""

    observation: StatePolicyObservation
    model_state: np.ndarray
    model_action: np.ndarray
    actions: tuple[NormalizedAction, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = _finite_vector(self.model_state, name="model_state", dim=STATE_BC_STATE_DIM)
        action = _finite_vector(self.model_action, name="model_action", dim=STATE_BC_ACTION_DIM)
        if not self.actions:
            raise ValueError("state policy prediction must contain at least one action.")
        object.__setattr__(self, "model_state", state)
        object.__setattr__(self, "model_action", action)
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "metadata", dict(self.metadata))


class StatePolicyAdapter:
    """Adapt one state policy without owning simulator execution."""

    def __init__(
        self,
        *,
        artifact_path: str | Path | None = None,
        model: StatePolicyModel | None = None,
        config: StatePolicyAdapterConfig | None = None,
    ) -> None:
        if (artifact_path is None) == (model is None):
            raise ValueError("provide exactly one of artifact_path or model.")
        self.config = StatePolicyAdapterConfig() if config is None else config
        self.observation_adapter = StatePolicyObservationAdapter(
            StatePolicyObservationConfig(
                active_sides=self.config.active_sides,
                policy_absolute_frame=self.config.policy_absolute_frame,
                runtime_pose_frame_contract=self.config.runtime_pose_frame_contract,
                expected_layout_hash=self.config.expected_layout_hash,
                metadata=self.config.metadata,
            )
        )
        self.layout = self.observation_adapter.layout
        self._v22 = V22DeploymentAdapter(self.layout)
        self.model = _NpzStatePolicy(artifact_path) if artifact_path is not None else model
        if not callable(getattr(self.model, "predict", None)):
            raise TypeError("model must expose a callable predict(state) method.")
        self.artifact_metadata = dict(getattr(self.model, "metadata", {}) or {})
        artifact_layout = self.artifact_metadata.get("layout_hash")
        if artifact_layout is not None and str(artifact_layout) not in STATE_BC_ACCEPTED_LAYOUT_HASHES:
            raise ValueError("state BC model and adapter layout hashes disagree.")
        artifact_frame = self.artifact_metadata.get("pose_frame")
        if artifact_frame is not None and self.config.policy_absolute_frame == POLICY_ABSOLUTE_FRAME_CANONICAL_DESK:
            if str(artifact_frame).strip().lower() not in {"desk_absolute", "desk"}:
                raise ValueError("state BC artifact pose frame is not desk_absolute.")

    @property
    def artifact_hash(self) -> str | None:
        value = getattr(self.model, "artifact_hash", None)
        return None if value is None else str(value)

    def reset(self) -> None:
        """Keep the adapter stateless; queue/cursor ownership stays lower."""

    def predict(self, observation: DeploymentObservation) -> StatePolicyPrediction:
        model_observation = self.observation_adapter.capture(observation)
        model_state = model_observation.state
        raw_action = self.model.predict(model_state)  # type: ignore[union-attr]
        model_action = _normalize_pose_quaternion(
            _finite_vector(raw_action, name="model_action", dim=STATE_BC_ACTION_DIM),
            name="model_action",
        )
        chunk = V22PolicyActionChunk(
            actions=model_action.reshape(1, STATE_BC_ACTION_DIM),
            layout_hash=self.layout.layout_hash,
            metadata={
                "state_bc_schema": "dexteleop.state_bc_experiment.v1",
                "state_bc_model_artifact_hash": self.artifact_hash,
                "policy_absolute_frame": self.config.policy_absolute_frame,
            },
        )
        normalized_actions = self._v22.to_legacy_normalized_actions(
            chunk,
            source=self.config.source,
        )
        actions: list[NormalizedAction] = []
        for normalized in normalized_actions:
            runtime_vector = convert_policy_absolute_vector_to_runtime_raw(
                normalized.vector,
                active_sides=self.config.active_sides,
                policy_absolute_frame=self.config.policy_absolute_frame,
                runtime_pose_frame_contract=self.config.runtime_pose_frame_contract,
            )
            metadata = {
                **dict(normalized.metadata),
                "state_bc_schema": "dexteleop.state_bc_experiment.v1",
                "state_bc_model_artifact_hash": self.artifact_hash,
                "state_bc_model_state_dim": STATE_BC_STATE_DIM,
                "state_bc_model_action_dim": STATE_BC_ACTION_DIM,
                "policy_absolute_frame": self.config.policy_absolute_frame,
                "action_pose_mode": "absolute",
            }
            actions.append(replace(normalized, vector=runtime_vector, metadata=metadata))
        metadata = {
            **dict(self.config.metadata),
            "state_bc_schema": "dexteleop.state_bc_experiment.v1",
            "policy_io_layout_hash": self.layout.layout_hash,
            "policy_absolute_frame": self.config.policy_absolute_frame,
            "state_bc_model_artifact_hash": self.artifact_hash,
        }
        return StatePolicyPrediction(
            observation=model_observation,
            model_state=model_state,
            model_action=model_action,
            actions=tuple(actions),
            metadata=metadata,
        )


__all__ = [
    "STATE_BC_LAYOUT_HASH_V1",
    "STATE_BC_MODEL_SCHEMA_V1",
    "StatePolicyAdapter",
    "StatePolicyAdapterConfig",
    "StatePolicyModel",
    "StatePolicyPrediction",
]

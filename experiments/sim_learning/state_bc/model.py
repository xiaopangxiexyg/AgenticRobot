"""Small dependency-light state-to-action behavior-cloning model.

The first baseline is intentionally a regularized linear map.  It is not
presented as the final learner; it is a deterministic probe that lets the
dataset schema, frame conversion, safety gate, and HITL lineage be tested
before introducing a heavyweight policy framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    STATE_BC_MODEL_SCHEMA_V1,
    STATE_BC_ACTION_DIM,
    STATE_BC_STATE_DIM,
    StateBCArtifactMetadata,
    StateBCNormalization,
    validate_action_vector,
    validate_state_vector,
)


def _matrix(value: Any, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}.")
    return array.astype(np.float32, copy=True)


def _vector(value: Any, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape ({size},), got {array.shape}.")
    return array.astype(np.float32, copy=True)


@dataclass(frozen=True)
class StateBCModel:
    """Serialized linear state policy with explicit input normalization."""

    weights: np.ndarray
    bias: np.ndarray
    normalization: StateBCNormalization
    artifact_metadata: StateBCArtifactMetadata
    fit_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = _matrix(
            self.weights,
            name="weights",
            shape=(STATE_BC_STATE_DIM, STATE_BC_ACTION_DIM),
        )
        bias = _vector(self.bias, name="bias", size=STATE_BC_ACTION_DIM)
        if not isinstance(self.normalization, StateBCNormalization):
            raise TypeError("normalization must be a StateBCNormalization.")
        if not isinstance(self.artifact_metadata, StateBCArtifactMetadata):
            raise TypeError("artifact_metadata must be StateBCArtifactMetadata.")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "fit_metrics", dict(self.fit_metrics))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.artifact_metadata.to_dict(),
            "normalization": self.normalization.to_dict(),
            "fit_metrics": dict(self.fit_metrics),
        }

    @property
    def artifact_hash(self) -> str:
        payload = json.dumps(
            self.metadata,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        # The model parameters are part of the artifact identity, not only the
        # human-readable manifest.
        import hashlib

        digest = hashlib.sha256()
        digest.update(payload)
        digest.update(np.ascontiguousarray(self.weights).tobytes())
        digest.update(np.ascontiguousarray(self.bias).tobytes())
        digest.update(np.ascontiguousarray(self.normalization.mean).tobytes())
        digest.update(np.ascontiguousarray(self.normalization.scale).tobytes())
        return f"sha256:{digest.hexdigest()}"

    def predict(self, state: Any) -> np.ndarray:
        value = validate_state_vector(state)
        normalized = self.normalization.transform(value)
        action = normalized @ self.weights + self.bias
        return validate_action_vector(action, name="predicted_action")

    def save(self, path: str | Path) -> Path:
        """Write one self-contained compressed artifact (no pickle/object arrays)."""

        target = Path(path).expanduser()
        if target.suffix != ".npz":
            target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata["artifact_hash"] = self.artifact_hash
        np.savez_compressed(
            target,
            weights=self.weights,
            bias=self.bias,
            input_mean=self.normalization.mean,
            input_scale=self.normalization.scale,
            metadata_json=np.asarray(
                json.dumps(
                    metadata,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "StateBCModel":
        source = Path(path).expanduser()
        with np.load(source, allow_pickle=False) as payload:
            required = {"weights", "bias", "input_mean", "input_scale", "metadata_json"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"state BC artifact is missing arrays: {missing}.")
            raw_metadata = payload["metadata_json"]
            metadata_text = str(raw_metadata.item() if raw_metadata.ndim == 0 else raw_metadata)
            metadata = json.loads(metadata_text)
            if not isinstance(metadata, Mapping):
                raise ValueError("state BC artifact metadata_json must be an object.")
            if str(metadata.get("schema")) != STATE_BC_MODEL_SCHEMA_V1:
                raise ValueError("unsupported state BC artifact schema.")
            artifact_metadata = StateBCArtifactMetadata(
                selector_id=str(metadata["selector_id"]),
                train_episode_count=int(metadata["train_episode_count"]),
                test_episode_count=int(metadata["test_episode_count"]),
                train_frame_count=int(metadata["train_frame_count"]),
                test_frame_count=int(metadata["test_frame_count"]),
                model_type=str(metadata.get("model_type", "linear_ridge_v1")),
                metadata=dict(metadata.get("metadata", {})),
            )
            normalization = StateBCNormalization(
                mean=payload["input_mean"],
                scale=payload["input_scale"],
            )
            model = cls(
                weights=payload["weights"],
                bias=payload["bias"],
                normalization=normalization,
                artifact_metadata=artifact_metadata,
                fit_metrics=dict(metadata.get("fit_metrics", {})),
            )
            declared_hash = metadata.get("artifact_hash")
            if declared_hash is not None and str(declared_hash) != model.artifact_hash:
                raise ValueError("state BC artifact hash does not match its contents.")
            return model


def fit_linear_state_bc(
    states: Any,
    actions: Any,
    *,
    artifact_metadata: StateBCArtifactMetadata,
    ridge: float = 1e-4,
    fit_metrics: Mapping[str, Any] | None = None,
) -> StateBCModel:
    """Fit a ridge-regularized one-step absolute state policy."""

    x = np.asarray(states, dtype=np.float32)
    y = np.asarray(actions, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != STATE_BC_STATE_DIM:
        raise ValueError(f"states must have shape [N, {STATE_BC_STATE_DIM}], got {x.shape}.")
    if y.ndim != 2 or y.shape != (x.shape[0], STATE_BC_ACTION_DIM):
        raise ValueError(
            f"actions must have shape [{x.shape[0]}, {STATE_BC_ACTION_DIM}], got {y.shape}."
        )
    if x.shape[0] < 1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("states/actions must contain at least one finite row.")
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be finite and non-negative.")

    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(standard_deviation > 1e-6, standard_deviation, 1.0).astype(np.float32)
    normalization = StateBCNormalization(mean=mean, scale=scale)
    normalized = (x - normalization.mean) / normalization.scale
    augmented = np.concatenate(
        (normalized.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    target = y.astype(np.float64)
    gram = augmented.T @ augmented
    gram += np.eye(gram.shape[0], dtype=np.float64) * ridge_value
    rhs = augmented.T @ target
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    weights = coefficients[:-1].astype(np.float32)
    bias = coefficients[-1].astype(np.float32)
    return StateBCModel(
        weights=weights,
        bias=bias,
        normalization=normalization,
        artifact_metadata=artifact_metadata,
        fit_metrics={} if fit_metrics is None else dict(fit_metrics),
    )


__all__ = ["StateBCModel", "fit_linear_state_bc"]

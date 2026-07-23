"""Fail-closed model I/O manifests for simulator and future real adapters.

The manifest is deliberately backend-neutral.  It does not load a checkpoint
or initialise hardware; it only proves that an artifact declares the same
observation/action contracts as the adapter that is about to call it.  A real
deployment adapter can use these checks before it crosses its own execution
boundary, while the MuJoCo pipeline can use the exact same receipt in a
side-effect-free dry run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .contracts import (
    LEARNING_CONTRACT_VERSION,
    ModelInputBinding,
    ObservationContract,
    PolicyActionBinding,
)
from .model_io import EncodedModelInput, encode_model_input


MODEL_MANIFEST_VERSION = "dexteleop.policy_model_manifest.v1"


def _required_text(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclass(frozen=True)
class PolicyModelManifest:
    """Artifact-declared contract identities and model dimensions.

    ``artifact_sha256`` is an identity supplied by the checkpoint/artifact
    producer.  This class intentionally does not open the artifact, so a
    caller can verify the file digest separately without importing a policy
    framework into ``sim.mujoco.learning``.
    """

    model_id: str
    artifact_sha256: str
    observation_contract_hash: str
    model_input_binding_hash: str
    action_binding_hash: str
    input_dim: int
    output_dim: int
    policy_class: str = "normal"
    manifest_version: str = MODEL_MANIFEST_VERSION
    learning_contract_version: str = LEARNING_CONTRACT_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _required_text(self.model_id, "model_id"))
        artifact_sha256 = _required_text(self.artifact_sha256, "artifact_sha256").lower()
        if len(artifact_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in artifact_sha256
        ):
            raise ValueError("artifact_sha256 must be a 64-character SHA-256 hex digest")
        object.__setattr__(self, "artifact_sha256", artifact_sha256)
        for name in (
            "observation_contract_hash",
            "model_input_binding_hash",
            "action_binding_hash",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if self.manifest_version != MODEL_MANIFEST_VERSION:
            raise ValueError(f"Unsupported model manifest version: {self.manifest_version!r}")
        if self.learning_contract_version != LEARNING_CONTRACT_VERSION:
            raise ValueError(
                "Unsupported learning contract version in model manifest: "
                f"{self.learning_contract_version!r}"
            )
        policy_class = _required_text(self.policy_class, "policy_class")
        if policy_class not in {"normal", "privileged_expert"}:
            raise ValueError(
                "policy_class must be 'normal' or 'privileged_expert'"
            )
        object.__setattr__(self, "policy_class", policy_class)
        input_dim = int(self.input_dim)
        output_dim = int(self.output_dim)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("model input_dim and output_dim must be positive")
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "output_dim", output_dim)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def validate(
        self,
        contract: ObservationContract,
        binding: ModelInputBinding,
        action_binding: PolicyActionBinding,
    ) -> None:
        """Reject a model before inference when any semantic identity drifts."""

        if self.observation_contract_hash != contract.semantic_content_hash:
            raise ValueError(
                "Model manifest observation contract mismatch: "
                f"manifest={self.observation_contract_hash}, "
                f"runtime={contract.semantic_content_hash}"
            )
        if self.model_input_binding_hash != binding.semantic_content_hash:
            raise ValueError(
                "Model manifest input binding mismatch: "
                f"manifest={self.model_input_binding_hash}, "
                f"runtime={binding.semantic_content_hash}"
            )
        if self.action_binding_hash != action_binding.semantic_content_hash:
            raise ValueError(
                "Model manifest action binding mismatch: "
                f"manifest={self.action_binding_hash}, "
                f"runtime={action_binding.semantic_content_hash}"
            )
        binding_dim = binding.expected_flat_dim(contract)
        if self.input_dim != binding_dim:
            raise ValueError(
                "Model manifest input dimension mismatch: "
                f"manifest={self.input_dim}, runtime={binding_dim}"
            )
        if self.output_dim != action_binding.output_dim:
            raise ValueError(
                "Model manifest output dimension mismatch: "
                f"manifest={self.output_dim}, runtime={action_binding.output_dim}"
            )
        if self.policy_class == "normal" and binding.allow_privileged:
            raise ValueError("A normal model manifest cannot admit privileged inputs")
        if self.policy_class == "privileged_expert" and not binding.allow_privileged:
            raise ValueError(
                "A privileged_expert model manifest requires an explicitly privileged binding"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/checkpoint-safe representation."""

        return {
            "manifest_version": self.manifest_version,
            "learning_contract_version": self.learning_contract_version,
            "model_id": self.model_id,
            "artifact_sha256": self.artifact_sha256,
            "observation_contract_hash": self.observation_contract_hash,
            "model_input_binding_hash": self.model_input_binding_hash,
            "action_binding_hash": self.action_binding_hash,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "policy_class": self.policy_class,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyModelManifest":
        """Parse a manifest without accepting missing contract identities."""

        if not isinstance(payload, Mapping):
            raise TypeError("model manifest must be a mapping")
        return cls(
            model_id=payload.get("model_id", ""),
            artifact_sha256=payload.get("artifact_sha256", ""),
            observation_contract_hash=payload.get("observation_contract_hash", ""),
            model_input_binding_hash=payload.get("model_input_binding_hash", ""),
            action_binding_hash=payload.get("action_binding_hash", ""),
            input_dim=payload.get("input_dim", 0),
            output_dim=payload.get("output_dim", 0),
            policy_class=payload.get("policy_class", "normal"),
            manifest_version=payload.get("manifest_version", ""),
            learning_contract_version=payload.get("learning_contract_version", ""),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class ModelDryRunReceipt:
    """Evidence that one model input/output exchange passed the manifest."""

    model_id: str
    observation_contract_hash: str
    model_input_binding_hash: str
    action_binding_hash: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_finite: bool
    output_finite: bool

    @property
    def accepted(self) -> bool:
        return self.input_finite and self.output_finite

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "observation_contract_hash": self.observation_contract_hash,
            "model_input_binding_hash": self.model_input_binding_hash,
            "action_binding_hash": self.action_binding_hash,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "input_finite": self.input_finite,
            "output_finite": self.output_finite,
            "accepted": self.accepted,
        }


def dry_run_model_io(
    manifest: PolicyModelManifest,
    contract: ObservationContract,
    binding: ModelInputBinding,
    action_binding: PolicyActionBinding,
    observation: Mapping[str, Any] | Any,
    model_output: Any,
) -> ModelDryRunReceipt:
    """Validate one side-effect-free model exchange.

    The function accepts a precomputed output rather than invoking a model;
    this keeps it usable for both a MuJoCo smoke policy and a real-model
    adapter.  Shapes are strict: a batched input requires a matching batched
    action output, so accidental ``squeeze`` operations cannot pass silently.
    """

    manifest.validate(contract, binding, action_binding)
    encoded: EncodedModelInput = encode_model_input(observation, contract, binding)
    output = np.asarray(model_output, dtype=np.float32)
    if output.ndim == 0:
        raise ValueError("model output must be a one-dimensional or batched array")
    if output.ndim == 1:
        if encoded.flat.ndim != 1:
            raise ValueError(
                "batched model input requires a batched model output with shape (N, action_dim)"
            )
        if output.shape != (manifest.output_dim,):
            raise ValueError(
                "model output dimension mismatch: "
                f"expected {(manifest.output_dim,)}, got {output.shape}"
            )
    elif output.ndim == 2:
        if encoded.flat.ndim != 2 or output.shape[0] != encoded.flat.shape[0]:
            raise ValueError(
                "model input/output batch shape mismatch: "
                f"input={encoded.flat.shape}, output={output.shape}"
            )
        if output.shape[1] != manifest.output_dim:
            raise ValueError(
                "model output dimension mismatch: "
                f"expected {manifest.output_dim}, got {output.shape[1]}"
            )
    else:
        raise ValueError("model output must have rank 1 or 2")
    input_finite = bool(np.all(np.isfinite(encoded.flat)))
    output_finite = bool(np.all(np.isfinite(output)))
    if not input_finite or not output_finite:
        raise ValueError("model dry run encountered non-finite input or output")
    return ModelDryRunReceipt(
        model_id=manifest.model_id,
        observation_contract_hash=encoded.observation_contract_hash,
        model_input_binding_hash=encoded.binding_hash,
        action_binding_hash=action_binding.semantic_content_hash,
        input_shape=tuple(int(size) for size in encoded.flat.shape),
        output_shape=tuple(int(size) for size in output.shape),
        input_finite=input_finite,
        output_finite=output_finite,
    )


__all__ = [
    "MODEL_MANIFEST_VERSION",
    "ModelDryRunReceipt",
    "PolicyModelManifest",
    "dry_run_model_io",
]

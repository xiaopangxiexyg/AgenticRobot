"""Pure provenance contracts for trained model-RTC runs and checkpoints.

These receipts intentionally stay outside the model-continuation data
contract.  The latter describes tensor/runtime semantics; these values prove
which dataset manifests, sampling specification, trainer and base checkpoint
were used by one concrete training run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .canonical_encoding import canonical_content_hash


MODEL_CONTINUATION_TRAINING_SOURCE_SCHEMA_V1 = (
    "dexteleop_model_continuation_training_source_v1"
)
MODEL_CONTINUATION_TRAINING_RUN_SCHEMA_V1 = (
    "dexteleop_model_continuation_training_run_v1"
)
MODEL_CONTINUATION_CHECKPOINT_RECEIPT_SCHEMA_V1 = (
    "dexteleop_model_continuation_checkpoint_receipt_v1"
)
MODEL_CONTINUATION_TRAINING_SOURCE_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_training_source.v1"
)
MODEL_CONTINUATION_TRAINING_RUN_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_training_run.v1"
)
MODEL_CONTINUATION_CHECKPOINT_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_checkpoint_receipt.v1"
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _sha256(value: object, *, name: str) -> str:
    normalized = _nonempty(value, name=name)
    if normalized.startswith("sha256:"):
        digest = normalized[7:]
    else:
        digest = normalized
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a canonical SHA-256 identity.")
    return f"sha256:{digest}"


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


@dataclass(frozen=True)
class ModelContinuationTrainingSourceBinding:
    """Immutable metadata/media-index identity for one training source."""

    source_id: str
    dataset_schema: str
    camera_keys: tuple[str, ...]
    dataset_identity_hash: str
    episode_manifest_hash: str
    media_index_hash: str
    schema: str = MODEL_CONTINUATION_TRAINING_SOURCE_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != MODEL_CONTINUATION_TRAINING_SOURCE_SCHEMA_V1:
            raise ValueError("Unsupported model-RTC training source schema.")
        source_id = _nonempty(self.source_id, name="source_id")
        dataset_schema = _nonempty(
            self.dataset_schema,
            name="dataset_schema",
        )
        camera_keys = tuple(
            dict.fromkeys(
                _nonempty(value, name="camera_keys entry")
                for value in self.camera_keys
            )
        )
        if not camera_keys:
            raise ValueError("camera_keys must contain at least one logical view.")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "dataset_schema", dataset_schema)
        object.__setattr__(self, "camera_keys", camera_keys)
        for name in (
            "dataset_identity_hash",
            "episode_manifest_hash",
            "media_index_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_id": self.source_id,
            "dataset_schema": self.dataset_schema,
            "camera_keys": list(self.camera_keys),
            "dataset_identity_hash": self.dataset_identity_hash,
            "episode_manifest_hash": self.episode_manifest_hash,
            "media_index_hash": self.media_index_hash,
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=MODEL_CONTINUATION_TRAINING_SOURCE_HASH_DOMAIN_V1,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ModelContinuationTrainingSourceBinding":
        expected = {
            "schema",
            "source_id",
            "dataset_schema",
            "camera_keys",
            "dataset_identity_hash",
            "episode_manifest_hash",
            "media_index_hash",
        }
        unknown = sorted(set(payload).difference(expected))
        if unknown:
            raise ValueError(
                f"Model-RTC training source contains unknown entries: {unknown}."
            )
        camera_keys = payload.get("camera_keys")
        if not isinstance(camera_keys, Sequence) or isinstance(
            camera_keys,
            (str, bytes),
        ):
            raise ValueError("camera_keys must be a sequence.")
        return cls(
            source_id=str(payload.get("source_id", "")),
            dataset_schema=str(payload.get("dataset_schema", "")),
            camera_keys=tuple(str(value) for value in camera_keys),
            dataset_identity_hash=str(
                payload.get("dataset_identity_hash", "")
            ),
            episode_manifest_hash=str(
                payload.get("episode_manifest_hash", "")
            ),
            media_index_hash=str(payload.get("media_index_hash", "")),
            schema=str(payload.get("schema", "")),
        )


@dataclass(frozen=True)
class ModelContinuationTrainingRunManifest:
    """Pre-run identity shared by every checkpoint produced by one run."""

    output_model_id: str
    policy_artifact_template_hash: str
    base_checkpoint_identity_hash: str
    sampling_spec_hash: str
    trainer_implementation_id: str
    trainer_parameters_hash: str
    train_config_hash: str
    source_bindings: tuple[ModelContinuationTrainingSourceBinding, ...]
    schema: str = MODEL_CONTINUATION_TRAINING_RUN_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != MODEL_CONTINUATION_TRAINING_RUN_SCHEMA_V1:
            raise ValueError("Unsupported model-RTC training run schema.")
        object.__setattr__(
            self,
            "output_model_id",
            _nonempty(self.output_model_id, name="output_model_id"),
        )
        object.__setattr__(
            self,
            "trainer_implementation_id",
            _nonempty(
                self.trainer_implementation_id,
                name="trainer_implementation_id",
            ),
        )
        for name in (
            "policy_artifact_template_hash",
            "base_checkpoint_identity_hash",
            "sampling_spec_hash",
            "trainer_parameters_hash",
            "train_config_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        sources = tuple(self.source_bindings)
        if not sources or any(
            not isinstance(item, ModelContinuationTrainingSourceBinding)
            for item in sources
        ):
            raise TypeError(
                "source_bindings must contain at least one training source."
            )
        source_ids = tuple(item.source_id for item in sources)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Training source_id values must be unique.")
        canonical_sources = tuple(
            sorted(sources, key=lambda item: item.source_id)
        )
        object.__setattr__(self, "source_bindings", canonical_sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "output_model_id": self.output_model_id,
            "policy_artifact_template_hash": (
                self.policy_artifact_template_hash
            ),
            "base_checkpoint_identity_hash": (
                self.base_checkpoint_identity_hash
            ),
            "sampling_spec_hash": self.sampling_spec_hash,
            "trainer_implementation_id": self.trainer_implementation_id,
            "trainer_parameters_hash": self.trainer_parameters_hash,
            "train_config_hash": self.train_config_hash,
            "source_bindings": [
                item.to_dict() for item in self.source_bindings
            ],
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=MODEL_CONTINUATION_TRAINING_RUN_HASH_DOMAIN_V1,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ModelContinuationTrainingRunManifest":
        expected = {
            "schema",
            "output_model_id",
            "policy_artifact_template_hash",
            "base_checkpoint_identity_hash",
            "sampling_spec_hash",
            "trainer_implementation_id",
            "trainer_parameters_hash",
            "train_config_hash",
            "source_bindings",
        }
        unknown = sorted(set(payload).difference(expected))
        if unknown:
            raise ValueError(
                f"Model-RTC training run contains unknown entries: {unknown}."
            )
        sources = payload.get("source_bindings")
        if not isinstance(sources, Sequence) or isinstance(
            sources,
            (str, bytes),
        ):
            raise ValueError("source_bindings must be a sequence.")
        if any(not isinstance(item, Mapping) for item in sources):
            raise ValueError(
                "Every source_bindings entry must be an object."
            )
        return cls(
            output_model_id=str(payload.get("output_model_id", "")),
            policy_artifact_template_hash=str(
                payload.get("policy_artifact_template_hash", "")
            ),
            base_checkpoint_identity_hash=str(
                payload.get("base_checkpoint_identity_hash", "")
            ),
            sampling_spec_hash=str(payload.get("sampling_spec_hash", "")),
            trainer_implementation_id=str(
                payload.get("trainer_implementation_id", "")
            ),
            trainer_parameters_hash=str(
                payload.get("trainer_parameters_hash", "")
            ),
            train_config_hash=str(payload.get("train_config_hash", "")),
            source_bindings=tuple(
                ModelContinuationTrainingSourceBinding.from_dict(item)
                for item in sources
            ),
            schema=str(payload.get("schema", "")),
        )


@dataclass(frozen=True)
class ModelContinuationCheckpointReceipt:
    """Bind one saved checkpoint to its immutable training-run manifest."""

    training_run_manifest_hash: str
    policy_artifact_hash: str
    model_id: str
    checkpoint_step: int
    schema: str = MODEL_CONTINUATION_CHECKPOINT_RECEIPT_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != MODEL_CONTINUATION_CHECKPOINT_RECEIPT_SCHEMA_V1:
            raise ValueError("Unsupported model-RTC checkpoint receipt schema.")
        object.__setattr__(
            self,
            "training_run_manifest_hash",
            _sha256(
                self.training_run_manifest_hash,
                name="training_run_manifest_hash",
            ),
        )
        object.__setattr__(
            self,
            "policy_artifact_hash",
            _sha256(self.policy_artifact_hash, name="policy_artifact_hash"),
        )
        object.__setattr__(
            self,
            "model_id",
            _nonempty(self.model_id, name="model_id"),
        )
        object.__setattr__(
            self,
            "checkpoint_step",
            _exact_nonnegative_int(
                self.checkpoint_step,
                name="checkpoint_step",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "training_run_manifest_hash": self.training_run_manifest_hash,
            "policy_artifact_hash": self.policy_artifact_hash,
            "model_id": self.model_id,
            "checkpoint_step": self.checkpoint_step,
            "semantic_content_hash": self.semantic_content_hash,
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "training_run_manifest_hash": (
                    self.training_run_manifest_hash
                ),
                "policy_artifact_hash": self.policy_artifact_hash,
                "model_id": self.model_id,
                "checkpoint_step": self.checkpoint_step,
            },
            domain=MODEL_CONTINUATION_CHECKPOINT_RECEIPT_HASH_DOMAIN_V1,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ModelContinuationCheckpointReceipt":
        expected = {
            "schema",
            "training_run_manifest_hash",
            "policy_artifact_hash",
            "model_id",
            "checkpoint_step",
            "semantic_content_hash",
        }
        unknown = sorted(set(payload).difference(expected))
        if unknown:
            raise ValueError(
                "Model-RTC checkpoint receipt contains unknown entries: "
                f"{unknown}."
            )
        receipt = cls(
            training_run_manifest_hash=str(
                payload.get("training_run_manifest_hash", "")
            ),
            policy_artifact_hash=str(
                payload.get("policy_artifact_hash", "")
            ),
            model_id=str(payload.get("model_id", "")),
            checkpoint_step=payload.get("checkpoint_step", -1),
            schema=str(payload.get("schema", "")),
        )
        declared_hash = payload.get("semantic_content_hash")
        if (
            declared_hash is not None
            and str(declared_hash) != receipt.semantic_content_hash
        ):
            raise ValueError(
                "Model-RTC checkpoint receipt semantic_content_hash "
                "does not match its content."
            )
        return receipt


__all__ = [
    "MODEL_CONTINUATION_CHECKPOINT_RECEIPT_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_CHECKPOINT_RECEIPT_SCHEMA_V1",
    "MODEL_CONTINUATION_TRAINING_RUN_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_TRAINING_RUN_SCHEMA_V1",
    "MODEL_CONTINUATION_TRAINING_SOURCE_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_TRAINING_SOURCE_SCHEMA_V1",
    "ModelContinuationCheckpointReceipt",
    "ModelContinuationTrainingRunManifest",
    "ModelContinuationTrainingSourceBinding",
]

"""Deterministic model-continuation training views over artifact actions.

This module owns an offline projection only. It does not read or rewrite a
dataset, normalize model inputs, create framework tensors, call a model, or
participate in an RTC runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_LOSS_ALL_VALID_V1,
    MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    normalize_model_continuation_contract,
)
from dataset_contracts.policy_artifact import PolicyArtifactContract
from dataset_contracts.pose_frames import PoseFrameContract
from dataset_contracts.v22 import DatasetLayoutBindingError
from robot_motion.relative_action import ACTION_POSE_MODE_ABSOLUTE

from .pose_chunk import (
    PoseActionChunkSample,
    pose_action_chunk_from_episode_rows,
)


MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1 = (
    "dexteleop_model_continuation_training_view_v1"
)
MODEL_CONTINUATION_TRAINING_VIEW_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_training_view.v1"
)
MODEL_CONTINUATION_ARTIFACT_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_artifact_binding.v1"
)
TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1 = (
    "dataset_interface.trajectory_overlap_teacher_v1"
)
_TEACHER_OVERLAP_PROJECTION_SPEC_V1 = {
    "schema": MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1,
    "source_kind": TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    "action_representation": "artifact_action_continuous_absolute_v1",
    "delay_selection": "explicit_generation_steps_v1",
    "prefix_selection": "contiguous_generation_prefix_v1",
    "unselected_prefix_fill": "zero_v1",
    "prefix_mask": "float32_one_is_conditioned_v1",
    "target_padding_mask": "bool_true_is_padding_v1",
    "loss_mask": "bool_true_contributes_to_loss_v1",
}
TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1 = "sha256:" + (
    canonical_content_hash(
        _TEACHER_OVERLAP_PROJECTION_SPEC_V1,
        domain="dexteleop.model_continuation_teacher_projection_spec.v1",
    )
)


class ModelContinuationViewError(DatasetLayoutBindingError):
    """Raised when a continuation training view cannot be derived exactly."""


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ModelContinuationViewError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ModelContinuationViewError(f"{name} must be non-empty.")
    return normalized


def _exact_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ModelContinuationViewError(f"{name} must be an integer.")
    return int(value)


def _artifact_binding_hash(artifact: PolicyArtifactContract) -> str:
    assert artifact.pose_action_chunk is not None
    assert artifact.pose_training_data is not None
    assert artifact.model_continuation is not None
    return "sha256:" + canonical_content_hash(
        {
            "policy_io_layout": artifact.policy_io_layout.to_dict(),
            "pose_action_chunk": artifact.pose_action_chunk.to_dict(),
            "pose_training_data": artifact.pose_training_data.to_dict(),
            "pose_frame_contract_hashes": list(
                artifact.pose_frame_contract_hashes
            ),
            "model_continuation": artifact.model_continuation.to_dict(),
        },
        domain=MODEL_CONTINUATION_ARTIFACT_BINDING_HASH_DOMAIN_V1,
    )


@dataclass(frozen=True)
class ModelContinuationTrainingSample:
    """One artifact-action prefix/target/mask view.

    `prefix_mask == 1` means that generation step is conditioned.
    `target_is_pad == True` means the target step is padding.
    `loss_mask == True` means the target step contributes to loss.
    """

    conditioning_prefix: np.ndarray
    prefix_mask: np.ndarray
    target_action: np.ndarray
    target_is_pad: np.ndarray
    loss_mask: np.ndarray
    forecast_delay_steps: int
    prefix_steps: int
    training_prefix_source_kind: str
    source_episode_id: str
    source_start_index: int
    artifact_binding_hash: str
    continuation_contract_hash: str
    layout_hash: str
    chunk_contract_hash: str
    pose_frame_contract_hash: str
    projection_implementation_id: str
    projection_parameters_hash: str
    loss_mask_semantics: str
    schema: str = MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1:
            raise ModelContinuationViewError(
                f"Unsupported continuation training view schema {self.schema!r}."
            )
        target = np.asarray(self.target_action, dtype=np.float32)
        prefix = np.asarray(self.conditioning_prefix, dtype=np.float32)
        prefix_mask = np.asarray(self.prefix_mask, dtype=np.float32)
        target_is_pad = np.asarray(self.target_is_pad, dtype=bool)
        loss_mask = np.asarray(self.loss_mask, dtype=bool)
        if target.ndim != 2 or target.shape[0] < 2 or target.shape[1] < 1:
            raise ModelContinuationViewError(
                "target_action must be 2D [horizon, continuous_action_dim]."
            )
        horizon = target.shape[0]
        if prefix.shape != target.shape:
            raise ModelContinuationViewError(
                "conditioning_prefix must match target_action shape."
            )
        if (
            prefix_mask.shape != (horizon,)
            or target_is_pad.shape != (horizon,)
            or loss_mask.shape != (horizon,)
        ):
            raise ModelContinuationViewError(
                "Continuation masks must each have shape [horizon]."
            )
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prefix)):
            raise ModelContinuationViewError(
                "Continuation action values must be finite."
            )

        delay = _exact_int(
            self.forecast_delay_steps,
            name="forecast_delay_steps",
        )
        prefix_steps = _exact_int(self.prefix_steps, name="prefix_steps")
        start = _exact_int(self.source_start_index, name="source_start_index")
        if delay < 0 or delay >= horizon:
            raise ModelContinuationViewError(
                "forecast_delay_steps must be within the prediction horizon."
            )
        if prefix_steps < 0 or prefix_steps > delay:
            raise ModelContinuationViewError(
                "prefix_steps must be between zero and forecast_delay_steps."
            )
        if start < 0:
            raise ModelContinuationViewError(
                "source_start_index must be non-negative."
            )

        expected_prefix_mask = np.zeros((horizon,), dtype=np.float32)
        expected_prefix_mask[:prefix_steps] = 1.0
        if not np.array_equal(prefix_mask, expected_prefix_mask):
            raise ModelContinuationViewError(
                "prefix_mask must be a contiguous one-prefix/zero-suffix."
            )
        valid_count = int(np.count_nonzero(~target_is_pad))
        expected_padding = np.arange(horizon) >= valid_count
        if valid_count < 1 or not np.array_equal(
            target_is_pad,
            expected_padding,
        ):
            raise ModelContinuationViewError(
                "target_is_pad must be a contiguous false-prefix/true-tail."
            )
        if delay >= valid_count:
            raise ModelContinuationViewError(
                "forecast_delay_steps must leave a non-padding executable suffix."
            )
        if np.any((prefix_mask > 0) & target_is_pad):
            raise ModelContinuationViewError(
                "Conditioning prefix and target padding cannot overlap."
            )

        selected = prefix_mask > 0
        if not np.array_equal(prefix[selected], target[selected]):
            raise ModelContinuationViewError(
                "Teacher prefix must equal selected artifact-action targets."
            )
        if np.any(prefix[~selected] != 0):
            raise ModelContinuationViewError(
                "Unselected conditioning prefix entries must be zero-filled."
            )
        if (
            self.loss_mask_semantics
            == MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1
        ):
            expected_loss = ~target_is_pad & ~selected
        elif self.loss_mask_semantics == MODEL_CONTINUATION_LOSS_ALL_VALID_V1:
            expected_loss = ~target_is_pad
        else:
            raise ModelContinuationViewError(
                "Unsupported loss_mask_semantics."
            )
        if not np.array_equal(loss_mask, expected_loss):
            raise ModelContinuationViewError(
                "loss_mask does not match its declared semantics."
            )

        source_kind = _nonempty(
            self.training_prefix_source_kind,
            name="training_prefix_source_kind",
        )
        if source_kind != TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1:
            raise ModelContinuationViewError(
                "D2 supports only trajectory_overlap_teacher_v1 provenance."
            )
        for name in (
            "source_episode_id",
            "artifact_binding_hash",
            "continuation_contract_hash",
            "layout_hash",
            "chunk_contract_hash",
            "pose_frame_contract_hash",
            "projection_implementation_id",
            "projection_parameters_hash",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        if (
            self.projection_implementation_id
            != TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1
            or self.projection_parameters_hash
            != TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1
        ):
            raise ModelContinuationViewError(
                "Training sample projection identity does not match D2."
            )

        object.__setattr__(self, "conditioning_prefix", prefix.copy())
        object.__setattr__(self, "prefix_mask", prefix_mask.copy())
        object.__setattr__(self, "target_action", target.copy())
        object.__setattr__(self, "target_is_pad", target_is_pad.copy())
        object.__setattr__(self, "loss_mask", loss_mask.copy())
        object.__setattr__(self, "forecast_delay_steps", delay)
        object.__setattr__(self, "prefix_steps", prefix_steps)
        object.__setattr__(self, "source_start_index", start)
        object.__setattr__(self, "training_prefix_source_kind", source_kind)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "conditioning_prefix": self.conditioning_prefix,
            "prefix_mask": self.prefix_mask,
            "target_action": self.target_action,
            "target_is_pad": self.target_is_pad,
            "loss_mask": self.loss_mask,
            "forecast_delay_steps": self.forecast_delay_steps,
            "prefix_steps": self.prefix_steps,
            "training_prefix_source_kind": self.training_prefix_source_kind,
            "source_episode_id": self.source_episode_id,
            "source_start_index": self.source_start_index,
            "artifact_binding_hash": self.artifact_binding_hash,
            "continuation_contract_hash": self.continuation_contract_hash,
            "layout_hash": self.layout_hash,
            "chunk_contract_hash": self.chunk_contract_hash,
            "pose_frame_contract_hash": self.pose_frame_contract_hash,
            "projection_implementation_id": self.projection_implementation_id,
            "projection_parameters_hash": self.projection_parameters_hash,
            "loss_mask_semantics": self.loss_mask_semantics,
        }

    @property
    def sample_hash(self) -> str:
        return "sha256:" + canonical_content_hash(
            self.semantic_payload(),
            domain=MODEL_CONTINUATION_TRAINING_VIEW_HASH_DOMAIN_V1,
        )

    def artifact_action_dict(self) -> dict[str, Any]:
        """Return copies without normalization or framework conversion."""

        return {
            "continuation.prefix": self.conditioning_prefix.copy(),
            "continuation.prefix_mask": self.prefix_mask.copy(),
            "action": self.target_action.copy(),
            "action_is_pad": self.target_is_pad.copy(),
            "action_loss_mask": self.loss_mask.copy(),
            "forecast_delay_steps": self.forecast_delay_steps,
            "training_prefix_source_kind": (
                self.training_prefix_source_kind
            ),
            "model_continuation_data_contract_hash": (
                self.continuation_contract_hash
            ),
            "model_continuation_training_sample_hash": self.sample_hash,
        }


def model_continuation_training_sample_from_pose_chunk(
    chunk: PoseActionChunkSample,
    artifact: PolicyArtifactContract,
    *,
    source_episode_id: str,
    source_start_index: int,
    forecast_delay_steps: int,
    prefix_steps: int,
    training_prefix_source_kind: str = (
        TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1
    ),
) -> ModelContinuationTrainingSample:
    """Project one absolute continuous pose chunk into the D2 teacher view."""

    if not isinstance(chunk, PoseActionChunkSample):
        raise TypeError("chunk must be a PoseActionChunkSample.")
    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be a PolicyArtifactContract.")
    continuation = artifact.model_continuation
    chunk_contract = artifact.pose_action_chunk
    if continuation is None:
        raise ModelContinuationViewError(
            "Policy artifact has no model continuation contract."
        )
    normalized_continuation = normalize_model_continuation_contract(
        continuation
    )
    runtime_contract = normalized_continuation.runtime
    training_contract = normalized_continuation.training
    if training_contract is None:
        raise ModelContinuationViewError(
            "Inference-only continuation has no training projection."
        )
    if chunk_contract is None:
        raise ModelContinuationViewError(
            "D2 requires an artifact-bound pose action chunk contract."
        )
    if (
        training_contract.projection_implementation_id
        != TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1
        or training_contract.projection_parameters_hash
        != TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1
    ):
        raise ModelContinuationViewError(
            "Artifact continuation projection binding does not match the D2 "
            "teacher-overlap implementation."
        )
    source_kind = str(training_prefix_source_kind).strip()
    if source_kind not in training_contract.prefix_source_kinds:
        raise ModelContinuationViewError(
            "Requested training prefix source is not declared by the artifact."
        )
    if source_kind != TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1:
        raise ModelContinuationViewError(
            "D2 cannot derive recorded-policy or synthetic provenance."
        )
    if chunk_contract.action_pose_mode != ACTION_POSE_MODE_ABSOLUTE:
        raise ModelContinuationViewError(
            "D2 supports only absolute artifact-action chunks."
        )
    if artifact.policy_io_layout.action_discrete_dim != 0:
        raise ModelContinuationViewError(
            "D2 supports only continuous artifact actions."
        )
    if chunk.action_discrete.shape != (
        runtime_contract.prediction_horizon_steps,
        0,
    ):
        raise ModelContinuationViewError(
            "D2 requires an empty discrete action matrix."
        )
    if chunk.layout_hash != artifact.layout_hash:
        raise ModelContinuationViewError(
            "Chunk layout hash does not match the policy artifact."
        )
    if chunk.chunk_contract_hash != chunk_contract.contract_hash:
        raise ModelContinuationViewError(
            "Chunk contract hash does not match the policy artifact."
        )
    if chunk.pose_frame_contract_hash not in artifact.pose_frame_contract_hashes:
        raise ModelContinuationViewError(
            "Chunk pose frame contract is not bound by the policy artifact."
        )
    if chunk.action.shape != (
        runtime_contract.prediction_horizon_steps,
        runtime_contract.model_action_dim,
    ):
        raise ModelContinuationViewError(
            "Chunk action shape does not match the continuation contract."
        )
    if not np.all(np.isfinite(chunk.action)):
        raise ModelContinuationViewError(
            "Artifact-action target contains NaN or infinity."
        )

    delay = _exact_int(
        forecast_delay_steps,
        name="forecast_delay_steps",
    )
    selected_prefix_steps = _exact_int(prefix_steps, name="prefix_steps")
    if not (
        runtime_contract.minimum_forecast_delay_steps
        <= delay
        <= runtime_contract.maximum_forecast_delay_steps
    ):
        raise ModelContinuationViewError(
            "forecast_delay_steps is outside the artifact-supported range."
        )
    if not (
        runtime_contract.minimum_prefix_steps
        <= selected_prefix_steps
        <= runtime_contract.maximum_prefix_steps
    ):
        raise ModelContinuationViewError(
            "prefix_steps is outside the artifact-supported range."
        )
    if selected_prefix_steps > delay:
        raise ModelContinuationViewError(
            "prefix_steps cannot exceed forecast_delay_steps."
        )
    if delay >= chunk.valid_action_count:
        raise ModelContinuationViewError(
            "forecast_delay_steps leaves no real executable target."
        )

    horizon = runtime_contract.prediction_horizon_steps
    prefix_mask = np.zeros((horizon,), dtype=np.float32)
    prefix_mask[:selected_prefix_steps] = 1.0
    conditioning_prefix = np.zeros_like(chunk.action, dtype=np.float32)
    conditioning_prefix[:selected_prefix_steps] = chunk.action[
        :selected_prefix_steps
    ]
    target_is_pad = chunk.action_is_pad.astype(bool, copy=True)
    if (
        training_contract.loss_mask_semantics
        == MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1
    ):
        loss_mask = ~target_is_pad & ~(prefix_mask > 0)
    else:
        loss_mask = ~target_is_pad

    return ModelContinuationTrainingSample(
        conditioning_prefix=conditioning_prefix,
        prefix_mask=prefix_mask,
        target_action=chunk.action,
        target_is_pad=target_is_pad,
        loss_mask=loss_mask,
        forecast_delay_steps=delay,
        prefix_steps=selected_prefix_steps,
        training_prefix_source_kind=source_kind,
        source_episode_id=source_episode_id,
        source_start_index=source_start_index,
        artifact_binding_hash=_artifact_binding_hash(artifact),
        continuation_contract_hash=(
            normalized_continuation.source_contract_hash
        ),
        layout_hash=chunk.layout_hash,
        chunk_contract_hash=chunk.chunk_contract_hash,
        pose_frame_contract_hash=chunk.pose_frame_contract_hash,
        projection_implementation_id=(
            training_contract.projection_implementation_id
        ),
        projection_parameters_hash=(
            training_contract.projection_parameters_hash
        ),
        loss_mask_semantics=training_contract.loss_mask_semantics,
    )


def model_continuation_training_sample_from_episode_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    artifact: PolicyArtifactContract,
    *,
    pose_frame_contract: PoseFrameContract,
    source_episode_id: str,
    start_index: int,
    forecast_delay_steps: int,
    prefix_steps: int,
    active_sides: Sequence[str] = ("left", "right"),
    training_prefix_source_kind: str = (
        TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1
    ),
) -> ModelContinuationTrainingSample:
    """Build a D2 sample from rows already scoped to exactly one episode."""

    if artifact.pose_action_chunk is None:
        raise ModelContinuationViewError(
            "D2 requires an artifact-bound pose action chunk contract."
        )
    chunk = pose_action_chunk_from_episode_rows(
        episode_rows,
        artifact.policy_io_layout,
        pose_frame_contract=pose_frame_contract,
        chunk_contract=artifact.pose_action_chunk,
        start_index=start_index,
        active_sides=active_sides,
    )
    return model_continuation_training_sample_from_pose_chunk(
        chunk,
        artifact,
        source_episode_id=source_episode_id,
        source_start_index=start_index,
        forecast_delay_steps=forecast_delay_steps,
        prefix_steps=prefix_steps,
        training_prefix_source_kind=training_prefix_source_kind,
    )


__all__ = [
    "MODEL_CONTINUATION_ARTIFACT_BINDING_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_TRAINING_VIEW_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1",
    "TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1",
    "TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1",
    "ModelContinuationTrainingSample",
    "ModelContinuationViewError",
    "model_continuation_training_sample_from_episode_rows",
    "model_continuation_training_sample_from_pose_chunk",
]
